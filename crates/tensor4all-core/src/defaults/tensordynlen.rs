use crate::defaults::DynIndex;
use crate::index_like::IndexLike;
use crate::index_ops::{common_ind_positions, prepare_contraction, prepare_contraction_pairs};
use crate::tensor_like::LinearizationOrder;
use crate::AnyScalar;
use anyhow::{Context, Result};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tenferro::{DType, DotGeneralConfig, Tensor as NativeTensor};
use tenferro_ad::EagerTensor;
use tenferro_einsum::eager_tensor::einsum_subscripts as eager_einsum_ad;
use tenferro_einsum::EinsumSubscripts;
use tensor4all_tensorbackend::{
    axpby_native_tensor, contract_native_tensor, default_eager_ctx,
    dense_native_tensor_from_col_major, diag_native_tensor_from_col_major,
    native_tensor_primal_to_dense_col_major, native_tensor_primal_to_diag_c64,
    native_tensor_primal_to_diag_f64, scale_native_tensor, storage_payload_native_read_input,
    storage_to_native_tensor, NativeTensorReadInput, TensorElement,
};
use tensor4all_tensorbackend::{Storage, StorageKind};

use super::contract::PairwiseContractionOptions;
use super::structured_contraction::{
    normalize_payload_read_for_roots, storage_from_payload_native, storage_payload_native,
    OperandLayout, StructuredContractionPlan, StructuredContractionSpec,
};

fn conjugate_eager(
    inner: &EagerTensor,
) -> std::result::Result<EagerTensor, Arc<dyn std::error::Error + Send + Sync + 'static>> {
    inner.conj().map_err(|source| Arc::new(source) as _)
}

#[derive(Debug, Default, Clone)]
struct PairwiseContractProfileEntry {
    calls: usize,
    total_time: Duration,
    total_bytes: usize,
}

/// Hermitian eigendecomposition of a rank-2 [`TensorDynLen`].
/// Eigenvectors are returned as a rank-2 tensor whose first index is the input
/// matrix row index and whose second index labels eigenvector columns. The
/// eigenvalues are detached primal values intended for nonsmooth selection
/// logic such as truncation cutoffs.
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// let row = DynIndex::new_dyn(2);
/// let col = DynIndex::new_dyn(2);
/// let matrix = TensorDynLen::from_dense(
///     vec![row.clone(), col],
///     vec![1.0_f64, 0.0, 0.0, 2.0],
/// ).unwrap();
/// let decomp = matrix.hermitian_eigendecomposition(1.0e-12).unwrap();
/// assert_eq!(decomp.eigenvalues, vec![1.0, 2.0]);
/// assert_eq!(
///     decomp.eigenvectors.indices(),
///     &[row, decomp.eigenvector_index.clone()]
/// );
/// ```
#[derive(Debug, Clone)]
pub struct TensorHermitianEigendecomposition {
    /// Real eigenvalues in backend Hermitian eigensolver order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvector matrix with one eigenvector in each column.
    pub eigenvectors: TensorDynLen,
    /// Index labeling the eigenvector columns.
    pub eigenvector_index: DynIndex,
}

thread_local! {
    static PAIRWISE_CONTRACT_PROFILE_STATE: RefCell<HashMap<&'static str, PairwiseContractProfileEntry>> =
        RefCell::new(HashMap::new());
}

fn pairwise_contract_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("T4A_PROFILE_PAIRWISE_CONTRACT").is_ok())
}

fn record_pairwise_contract_profile(section: &'static str, elapsed: Duration) {
    if !pairwise_contract_profile_enabled() {
        return;
    }
    PAIRWISE_CONTRACT_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(section).or_default();
        entry.calls += 1;
        entry.total_time += elapsed;
    });
}

fn record_pairwise_contract_profile_bytes(section: &'static str, bytes: usize) {
    if !pairwise_contract_profile_enabled() {
        return;
    }
    PAIRWISE_CONTRACT_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(section).or_default();
        entry.total_bytes += bytes;
    });
}

fn profile_pairwise_contract_section<T>(section: &'static str, f: impl FnOnce() -> T) -> T {
    if !pairwise_contract_profile_enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    record_pairwise_contract_profile(section, started.elapsed());
    result
}

/// Reset the aggregated pairwise `TensorDynLen` contraction profile.
pub fn reset_pairwise_contract_profile() {
    PAIRWISE_CONTRACT_PROFILE_STATE.with(|state| state.borrow_mut().clear());
}

/// Print and clear the aggregated pairwise `TensorDynLen` contraction profile.
pub fn print_and_reset_pairwise_contract_profile() {
    if !pairwise_contract_profile_enabled() {
        return;
    }
    PAIRWISE_CONTRACT_PROFILE_STATE.with(|state| {
        let mut entries: Vec<_> = state
            .borrow()
            .iter()
            .map(|(section, entry)| (*section, entry.clone()))
            .collect();
        state.borrow_mut().clear();
        entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));

        eprintln!("=== TensorDynLen pairwise contract profile ===");
        for (section, entry) in entries {
            let per_call_us = if entry.calls == 0 {
                0.0
            } else {
                entry.total_time.as_secs_f64() * 1.0e6 / entry.calls as f64
            };
            eprintln!(
                "{section}: calls={} total={:.6}ms per_call={:.3}us bytes={}",
                entry.calls,
                entry.total_time.as_secs_f64() * 1.0e3,
                per_call_us,
                entry.total_bytes,
            );
        }
    });
}

fn native_tensor_profile_bytes(native: &NativeTensor) -> usize {
    let element_size = match native.dtype() {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::C32 => 8,
        DType::C64 => 16,
        DType::I32 => 4,
        DType::I64 => 8,
        DType::Bool => 1,
    };
    native.shape().iter().product::<usize>() * element_size
}

/// Trait for scalar types that can generate random values from a standard
/// normal distribution.
/// This enables the generic [`TensorDynLen::random`] constructor.
pub trait RandomScalar: TensorElement {
    /// Generate a random value from the standard normal distribution.
    fn random_value<R: Rng>(rng: &mut R) -> Self;
}

impl RandomScalar for f64 {
    fn random_value<R: Rng>(rng: &mut R) -> Self {
        StandardNormal.sample(rng)
    }
}

impl RandomScalar for Complex64 {
    fn random_value<R: Rng>(rng: &mut R) -> Self {
        Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng))
    }
}

/// Compute the permutation array from original indices to new indices.
/// This function finds the mapping from new indices to original indices by
/// matching index IDs. The result is a permutation array `perm` such that
/// `new_indices[i]` corresponds to `original_indices[perm[i]]`.
/// # Arguments
/// * `original_indices` - The original indices in their current order
/// * `new_indices` - The desired new indices order (must be a permutation of original_indices)
/// # Returns
/// A `Vec<usize>` representing the permutation: `perm[i]` is the position in
/// `original_indices` of the index that should be at position `i` in `new_indices`.
/// # Errors
/// Returns an error when `new_order` contains indices not present in
/// `original` (a missing-index failure) or the two lists differ in length
/// (a length mismatch).
/// # Example
/// ```
/// use tensor4all_core::tensor::compute_permutation_from_indices;
/// use tensor4all_core::DynIndex;
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let original = vec![i.clone(), j.clone()];
/// let new_order = vec![j.clone(), i.clone()];
/// let perm = compute_permutation_from_indices(&original, &new_order).unwrap();
/// assert_eq!(perm, vec![1, 0]);  // j is at position 1, i is at position 0
/// ```
pub fn compute_permutation_from_indices(
    original_indices: &[DynIndex],
    new_indices: &[DynIndex],
) -> Result<Vec<usize>> {
    anyhow::ensure!(
        new_indices.len() == original_indices.len(),
        "new_indices length must match original_indices length"
    );

    let mut perm = Vec::with_capacity(new_indices.len());
    let mut used = std::collections::HashSet::new();

    for new_idx in new_indices {
        // Find the position of this index in the original indices
        // DynIndex implements Eq, so we can compare directly
        let pos = original_indices
            .iter()
            .position(|old_idx| old_idx == new_idx)
            .ok_or_else(|| {
                anyhow::anyhow!("new_indices must be a permutation of original_indices")
            })?;

        anyhow::ensure!(used.insert(pos), "duplicate index in new_indices");
        perm.push(pos);
    }

    Ok(perm)
}

/// Compact structured payload kept in the authoritative eager representation.
/// The payload may use any supported eager dtype (`f32`, `f64`, `c32`, or
/// `c64`) and may be either tracked or untracked. Tracking is a property of
/// `payload`, never of the presence of this metadata container.
#[derive(Clone)]
pub(crate) struct StructuredPayload {
    payload: Arc<EagerTensor>,
    payload_dims: Vec<usize>,
    axis_classes: Vec<usize>,
}

/// Error returned when [`TensorDynLen::storage`] or
/// [`TensorDynLen::to_storage`] cannot produce a compact `f64`/`Complex64`
/// storage snapshot from the authoritative payload.
/// Backend diagnostics remain available through [`std::error::Error::source`]
/// instead of being erased into a display string. The error is cloneable so a
/// deferred failure can be retained by cloned tensors without rebuilding a
/// detached primal value.
/// # Examples
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_core::TensorStorageError;
/// let error = TensorStorageError::Materialization {
///     source: Arc::new(std::io::Error::other("backend unavailable")),
/// };
/// assert!(error.source().is_some());
/// assert!(error.to_string().contains("backend unavailable"));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorStorageError {
    /// The eager or structured payload could not be converted to compact storage.
    #[error("failed to materialize TensorDynLen storage: {source}")]
    Materialization {
        /// Original diagnostic returned by the backend or storage conversion seam.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// An eager payload uses a scalar dtype that compact [`Storage`] cannot hold.
    #[error(
        "compact TensorDynLen storage does not support dtype {dtype}; the eager payload remains authoritative"
    )]
    UnsupportedDtype {
        /// Native scalar dtype retained by the eager representation.
        dtype: &'static str,
    },
    /// An eager conjugation operation failed and was deferred by the infallible
    /// [`TensorDynLen::conj`] API.
    #[error("failed to conjugate TensorDynLen storage: {source}")]
    Conjugation {
        /// Original diagnostic returned by the eager AD backend.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

/// Errors returned by the fallible numerical and comparison methods on
/// [`TensorDynLen`].
/// The enum is intentionally owned by `tensor4all-core`: callers can match
/// storage, shape, scalar, subtraction, and invalid-value failures without
/// depending on the internal `anyhow` plumbing. Wrapped backend diagnostics
/// retain their complete [`std::error::Error::source`] chain.
/// # Examples
/// ```
/// use tensor4all_core::TensorDynLenError;
/// let error = TensorDynLenError::NaNInput {
///     operation: "norm_squared",
/// };
/// assert!(error.to_string().contains("NaN"));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorDynLenError {
    /// Compact storage or deferred storage materialization failed.
    #[error("TensorDynLen storage operation failed: {source}")]
    Storage {
        /// Original storage diagnostic, including its backend source chain.
        #[source]
        source: TensorStorageError,
    },
    /// A native eager payload could not be materialized for a numerical operation.
    #[error("TensorDynLen materialization failed: {source}")]
    Materialization {
        /// Original backend or eager-runtime diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// A rank-zero scalar could not be extracted from a reduction result.
    #[error("TensorDynLen scalar extraction failed: {source}")]
    ScalarExtraction {
        /// Original scalar-wrapper or backend diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// The reduction result has a scalar dtype that this real-valued operation
    /// cannot interpret.
    #[error("TensorDynLen scalar type mismatch: expected {expected}, got {actual}")]
    ScalarTypeMismatch {
        /// Scalar dtype required by the operation.
        expected: &'static str,
        /// Scalar dtype returned by the reduction.
        actual: String,
    },
    /// Tensor shapes or index spaces cannot be aligned for comparison.
    #[error("TensorDynLen shape mismatch during {operation}: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Operation that attempted the alignment.
        operation: &'static str,
        /// Expected index/dimension description.
        expected: String,
        /// Actual index/dimension description.
        actual: String,
    },
    /// Tensor subtraction failed while evaluating a comparison.
    #[error("TensorDynLen subtraction failed: {source}")]
    Subtraction {
        /// Original arithmetic or backend diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// An input contained a NaN and the operation rejected it rather than
    /// silently converting it to zero.
    #[error("TensorDynLen {operation} received NaN input")]
    NaNInput {
        /// Numerical operation that observed the NaN.
        operation: &'static str,
    },
    /// A comparison tolerance was NaN, infinite, or negative.
    #[error("TensorDynLen tolerance {name} is invalid: {value}")]
    InvalidTolerance {
        /// Name of the invalid tolerance.
        name: &'static str,
        /// Supplied tolerance value.
        value: f64,
    },
    /// Another eager tensor operation failed while preparing a comparison.
    #[error("TensorDynLen {operation} failed: {source}")]
    Operation {
        /// Name of the eager operation that failed.
        operation: &'static str,
        /// Original backend or tensor diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

impl From<anyhow::Error> for TensorDynLenError {
    fn from(source: anyhow::Error) -> Self {
        Self::operation("TensorDynLen", source)
    }
}

impl TensorDynLenError {
    fn boxed(error: anyhow::Error) -> Arc<dyn std::error::Error + Send + Sync + 'static> {
        Arc::from(error.into_boxed_dyn_error())
    }

    fn materialization(error: anyhow::Error) -> Self {
        Self::Materialization {
            source: Self::boxed(error),
        }
    }

    fn scalar_extraction(error: anyhow::Error) -> Self {
        Self::ScalarExtraction {
            source: Self::boxed(error),
        }
    }

    fn operation(operation: &'static str, error: anyhow::Error) -> Self {
        Self::Operation {
            operation,
            source: Self::boxed(error),
        }
    }
}

#[derive(Clone)]
pub(crate) enum TensorDynLenStorage {
    Materialized(Arc<Storage>),
    Eager {
        inner: Arc<EagerTensor>,
        axis_classes: Vec<usize>,
    },
    /// One authoritative compact eager payload and its logical layout.
    Compact(Arc<StructuredPayload>),
    /// A storage representation whose eager operation failed before the
    /// infallible tensor API could return an error.
    Deferred {
        source: Box<Self>,
        error: Arc<TensorStorageError>,
    },
}

impl TensorDynLenStorage {
    fn from_storage(storage: Arc<Storage>) -> Self {
        Self::Materialized(storage)
    }

    fn from_eager_dense(inner: EagerTensor, rank: usize) -> Self {
        Self::Eager {
            inner: Arc::new(inner),
            axis_classes: TensorDynLen::dense_axis_classes(rank),
        }
    }

    fn eager(&self) -> Option<&EagerTensor> {
        match self {
            Self::Materialized(_) => None,
            Self::Eager { inner, .. } => Some(inner.as_ref()),
            Self::Compact(payload) => Some(payload.payload.as_ref()),
            Self::Deferred { source, .. } => source.eager(),
        }
    }

    fn deferred_error(&self) -> Option<&TensorStorageError> {
        match self {
            Self::Deferred { error, .. } => Some(error.as_ref()),
            _ => None,
        }
    }

    fn with_deferred_error(self, error: TensorStorageError) -> Self {
        if self.deferred_error().is_some() {
            self
        } else {
            Self::Deferred {
                source: Box::new(self),
                error: Arc::new(error),
            }
        }
    }

    fn axis_classes(&self) -> &[usize] {
        match self {
            Self::Materialized(storage) => storage.axis_classes(),
            Self::Eager { axis_classes, .. } => axis_classes,
            Self::Compact(payload) => &payload.axis_classes,
            Self::Deferred { source, .. } => source.axis_classes(),
        }
    }

    fn payload_dims(&self) -> &[usize] {
        match self {
            Self::Materialized(storage) => storage.payload_dims(),
            Self::Eager { inner, .. } => inner.data().shape(),
            Self::Compact(payload) => &payload.payload_dims,
            Self::Deferred { source, .. } => source.payload_dims(),
        }
    }

    fn payload_strides_vec(&self) -> Vec<isize> {
        match self {
            Self::Materialized(storage) => storage.payload_strides().to_vec(),
            Self::Eager { inner, .. } => {
                TensorDynLen::col_major_strides(inner.data().shape()).unwrap_or_default()
            }
            Self::Compact(payload) => {
                TensorDynLen::col_major_strides(&payload.payload_dims).unwrap_or_default()
            }
            Self::Deferred { source, .. } => source.payload_strides_vec(),
        }
    }

    fn is_f64(&self) -> bool {
        match self {
            Self::Materialized(storage) => storage.is_f64(),
            Self::Eager { inner, .. } => inner.data().dtype() == DType::F64,
            Self::Compact(payload) => payload.payload.data().dtype() == DType::F64,
            Self::Deferred { source, .. } => source.is_f64(),
        }
    }

    fn is_c64(&self) -> bool {
        match self {
            Self::Materialized(storage) => storage.is_c64(),
            Self::Eager { inner, .. } => inner.data().dtype() == DType::C64,
            Self::Compact(payload) => payload.payload.data().dtype() == DType::C64,
            Self::Deferred { source, .. } => source.is_c64(),
        }
    }

    fn dtype(&self) -> Option<DType> {
        match self {
            Self::Materialized(storage) => Some(if storage.is_c64() {
                DType::C64
            } else {
                DType::F64
            }),
            Self::Eager { inner, .. } => Some(inner.data().dtype()),
            Self::Compact(payload) => Some(payload.payload.data().dtype()),
            Self::Deferred { source, .. } => source.dtype(),
        }
    }

    fn is_complex(&self) -> bool {
        match self {
            Self::Materialized(storage) => storage.is_complex(),
            Self::Eager { inner, .. } => matches!(inner.data().dtype(), DType::C32 | DType::C64),
            Self::Compact(payload) => {
                matches!(payload.payload.data().dtype(), DType::C32 | DType::C64)
            }
            Self::Deferred { source, .. } => source.is_complex(),
        }
    }

    fn is_diag(&self) -> bool {
        match self {
            Self::Materialized(storage) => storage.is_diag(),
            Self::Eager { axis_classes, .. } => TensorDynLen::is_diag_axis_classes(axis_classes),
            Self::Compact(payload) => TensorDynLen::is_diag_axis_classes(&payload.axis_classes),
            Self::Deferred { source, .. } => source.is_diag(),
        }
    }

    fn storage_kind(&self) -> StorageKind {
        match self {
            Self::Materialized(storage) => storage.storage_kind(),
            Self::Eager { axis_classes, .. } => {
                if axis_classes.iter().copied().eq(0..axis_classes.len()) {
                    StorageKind::Dense
                } else if TensorDynLen::is_diag_axis_classes(axis_classes) {
                    StorageKind::Diagonal
                } else {
                    StorageKind::Structured
                }
            }
            Self::Compact(payload) => {
                if payload
                    .axis_classes
                    .iter()
                    .copied()
                    .eq(0..payload.axis_classes.len())
                {
                    StorageKind::Dense
                } else if TensorDynLen::is_diag_axis_classes(&payload.axis_classes) {
                    StorageKind::Diagonal
                } else {
                    StorageKind::Structured
                }
            }
            Self::Deferred { source, .. } => source.storage_kind(),
        }
    }

    fn materialize(
        &self,
        logical_rank: usize,
    ) -> std::result::Result<Arc<Storage>, TensorStorageError> {
        match self {
            Self::Materialized(storage) => Ok(Arc::clone(storage)),
            Self::Eager {
                inner,
                axis_classes,
            } => {
                let dtype = inner.data().dtype();
                if matches!(dtype, DType::F32 | DType::C32) {
                    return Err(TensorStorageError::UnsupportedDtype {
                        dtype: TensorDynLen::dtype_name(dtype),
                    });
                }
                TensorDynLen::storage_from_native_with_axis_classes(
                    inner.data(),
                    axis_classes,
                    logical_rank,
                )
                .map(Arc::new)
                .map_err(|source| TensorStorageError::Materialization {
                    source: Arc::from(source.into_boxed_dyn_error()),
                })
            }
            Self::Compact(payload) => {
                let dtype = payload.payload.data().dtype();
                if matches!(dtype, DType::F32 | DType::C32) {
                    return Err(TensorStorageError::UnsupportedDtype {
                        dtype: TensorDynLen::dtype_name(dtype),
                    });
                }
                TensorDynLen::storage_from_native_with_axis_classes(
                    payload.payload.data(),
                    &payload.axis_classes,
                    logical_rank,
                )
                .map(Arc::new)
                .map_err(|source| TensorStorageError::Materialization {
                    source: Arc::from(source.into_boxed_dyn_error()),
                })
            }
            Self::Deferred { error, .. } => Err((**error).clone()),
        }
    }

    fn scale_eager_payload(&self, scalar: &AnyScalar) -> Result<Self> {
        let (payload, payload_dims, axis_classes) = match self {
            Self::Materialized(storage) => {
                let native = if storage.is_f64() {
                    let values = storage
                        .payload_f64_col_major_vec()
                        .map_err(anyhow::Error::new)?;
                    dense_native_tensor_from_col_major(&values, storage.payload_dims())?
                } else {
                    let values = storage
                        .payload_c64_col_major_vec()
                        .map_err(anyhow::Error::new)?;
                    dense_native_tensor_from_col_major(&values, storage.payload_dims())?
                };
                (
                    EagerTensor::from_tensor_in(native, default_eager_ctx()?),
                    storage.payload_dims().to_vec(),
                    storage.axis_classes().to_vec(),
                )
            }
            Self::Eager {
                inner,
                axis_classes,
            } => (
                (**inner).clone(),
                inner.data().shape().to_vec(),
                axis_classes.clone(),
            ),
            Self::Compact(compact) => (
                (*compact.payload).clone(),
                compact.payload_dims.clone(),
                compact.axis_classes.clone(),
            ),
            Self::Deferred { error, .. } => return Err(anyhow::Error::new((**error).clone())),
        };
        let scalar_inner = scalar.as_tensor()?.try_materialized_inner()?;
        let target_dtype =
            TensorDynLen::scale_target_dtype(payload.data().dtype(), scalar_inner.data().dtype())?;
        let payload = if payload.data().dtype() == target_dtype {
            payload
        } else {
            payload.convert(target_dtype)?
        };
        let scalar_inner = if scalar_inner.data().dtype() == target_dtype {
            scalar_inner.clone()
        } else {
            scalar_inner.convert(target_dtype)?
        };
        let scaled = if payload.data().shape().is_empty() {
            payload.mul(&scalar_inner)?
        } else {
            let subscripts = TensorDynLen::scale_subscripts(payload.data().shape().len())?;
            eager_einsum_ad(&[&payload, &scalar_inner], &subscripts)?
        };
        match self {
            Self::Eager { .. } => Ok(Self::Eager {
                inner: Arc::new(scaled),
                axis_classes,
            }),
            Self::Compact(_) | Self::Materialized(_) => {
                Ok(Self::Compact(Arc::new(StructuredPayload {
                    payload: Arc::new(scaled),
                    payload_dims,
                    axis_classes,
                })))
            }
            Self::Deferred { error, .. } => Err(anyhow::Error::new((**error).clone())),
        }
    }

    fn conjugate_with<F>(&self, conjugate: &F) -> std::result::Result<Self, TensorStorageError>
    where
        F: Fn(
            &EagerTensor,
        ) -> std::result::Result<
            EagerTensor,
            Arc<dyn std::error::Error + Send + Sync + 'static>,
        >,
    {
        match self {
            Self::Materialized(storage) => Ok(Self::Materialized(Arc::new(storage.conj()))),
            Self::Eager {
                inner,
                axis_classes,
            } => conjugate(inner)
                .map(|conjugated| Self::Eager {
                    inner: Arc::new(conjugated),
                    axis_classes: axis_classes.clone(),
                })
                .map_err(|source| TensorStorageError::Conjugation { source }),
            Self::Compact(payload) => conjugate(payload.payload.as_ref())
                .map(|conjugated| {
                    Self::Compact(Arc::new(StructuredPayload {
                        payload: Arc::new(conjugated),
                        payload_dims: payload.payload_dims.clone(),
                        axis_classes: payload.axis_classes.clone(),
                    }))
                })
                .map_err(|source| TensorStorageError::Conjugation { source }),
            Self::Deferred { error, .. } => Err((**error).clone()),
        }
    }

    fn sum_scalar(&self) -> Result<AnyScalar> {
        match self {
            Self::Materialized(storage) => {
                if storage.is_f64() {
                    Ok(AnyScalar::new_real(storage.sum::<f64>()))
                } else {
                    let value = storage.sum::<Complex64>();
                    Ok(AnyScalar::new_complex(value.re, value.im))
                }
            }
            Self::Eager { inner, .. } => TensorDynLen::native_sum_scalar(inner.data()),
            Self::Compact(payload) => TensorDynLen::native_sum_scalar(payload.payload.data()),
            Self::Deferred { error, .. } => Err(anyhow::Error::new((**error).clone())),
        }
    }

    fn nonfinite_flags(&self) -> Result<(bool, bool)> {
        match self {
            Self::Materialized(storage) => Ok(storage.payload_nonfinite_flags()),
            Self::Eager { inner, .. } => TensorDynLen::native_nonfinite_flags(inner.data()),
            Self::Compact(payload) => TensorDynLen::native_nonfinite_flags(payload.payload.data()),
            Self::Deferred { error, .. } => Err(anyhow::Error::new((**error).clone())),
        }
    }

    fn payload_value_at(&self, payload_coords: &[usize]) -> Result<Complex64> {
        match self {
            Self::Materialized(storage) => storage
                .scalar_at(payload_coords)
                .map(Complex64::from)
                .map_err(anyhow::Error::new),
            Self::Eager { inner, .. } => {
                TensorDynLen::native_complex_payload_value_at(inner.data(), payload_coords)
            }
            Self::Compact(payload) => TensorDynLen::native_complex_payload_value_at(
                payload.payload.data(),
                payload_coords,
            ),
            Self::Deferred { error, .. } => Err(anyhow::Error::new((**error).clone())),
        }
    }

    fn for_each_payload_value(&self, mut f: impl FnMut(Complex64)) -> Result<()> {
        let payload_dims = self.payload_dims();
        let payload_len = checked_product(payload_dims)?;
        let mut payload_coords = vec![0usize; payload_dims.len()];
        for _ in 0..payload_len {
            f(self.payload_value_at(&payload_coords)?);
            let mut carry = true;
            for (coordinate, &dim) in payload_coords.iter_mut().zip(payload_dims.iter()) {
                if !carry {
                    break;
                }
                *coordinate += 1;
                if *coordinate == dim {
                    *coordinate = 0;
                } else {
                    carry = false;
                }
            }
        }
        Ok(())
    }

    fn compact_payload(&self) -> Option<&StructuredPayload> {
        match self {
            Self::Compact(payload) => Some(payload.as_ref()),
            Self::Deferred { source, .. } => source.compact_payload(),
            _ => None,
        }
    }
}

/// Errors returned when constructing a compact copy-selector tensor.
/// A copy-selector has logical values
/// `scale * delta(left, right) * delta(site, selected_value)` and is used to
/// carry a bond through a fixed physical site without dense bond-squared storage.
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, StructuredSelectorError, TensorDynLen};
/// let left = DynIndex::new_dyn(2);
/// let site = DynIndex::new_dyn(3);
/// let right = DynIndex::new_dyn(4);
/// let error = TensorDynLen::from_copy_selector(left, site, right, 1, 1.0_f64)
///     .unwrap_err();
/// assert!(matches!(error, StructuredSelectorError::BondDimensionMismatch { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum StructuredSelectorError {
    /// The two logical copy axes have different dimensions.
    #[error("copy-selector bond dimensions differ: left={left}, right={right}")]
    BondDimensionMismatch {
        /// Dimension of the left copy axis.
        left: usize,
        /// Dimension of the right copy axis.
        right: usize,
    },
    /// One of the logical axes has dimension zero.
    #[error("copy-selector {axis} dimension must be positive")]
    ZeroDimension {
        /// Name of the zero-dimensional axis.
        axis: &'static str,
    },
    /// The selected physical coordinate is outside the site dimension.
    #[error("selected site value {value} is outside 0..{site_dim}")]
    SelectedValueOutOfBounds {
        /// Requested zero-based physical coordinate.
        value: usize,
        /// Dimension of the physical site.
        site_dim: usize,
    },
    /// The compact payload element count cannot be represented by `usize`.
    #[error("copy-selector payload size overflows usize for dimensions {bond_dim} x {site_dim}")]
    PayloadSizeOverflow {
        /// Dimension shared by the copy axes.
        bond_dim: usize,
        /// Dimension of the physical site.
        site_dim: usize,
    },
    /// A compact payload stride cannot be represented by `isize`.
    #[error("copy-selector bond stride {bond_dim} exceeds isize::MAX")]
    StrideOverflow {
        /// Bond dimension that could not be converted to a stride.
        bond_dim: usize,
    },
    /// Reserving the compact payload failed.
    #[error("could not allocate copy-selector payload with {elements} elements")]
    AllocationFailed {
        /// Number of compact payload elements requested.
        elements: usize,
    },
    /// Backend structured-storage validation failed.
    #[error("invalid copy-selector storage: {message}")]
    InvalidStorage {
        /// Diagnostic returned by structured-storage validation.
        message: String,
    },
}

/// Dynamic-rank tensor with structured payload storage -- the central data type
/// of tensor4all.
/// `TensorDynLen` stores a logical multi-dimensional tensor of supported scalar
/// values (`f32`, `f64`, `Complex32`, or `Complex64`) together with a list of
/// [`DynIndex`] labels. `f64`/`Complex64` tensors may use compact [`Storage`]
/// snapshots; `f32`/`Complex32` tensors retain an eager payload as the
/// authoritative representation because compact storage supports only the
/// 64-bit dtypes. The logical layout may be dense, diagonal, or explicitly
/// structured. The indices carry unique
/// identities (UUIDs) so that contraction, addition, and other binary
/// operations can automatically match legs by identity rather than position.
/// # Key Operations
/// | Operation | Method |
/// |-----------|--------|
/// | Create from data | [`from_dense`](Self::from_dense), [`from_diag`](Self::from_diag), [`zeros`](Self::zeros) |
/// | Extract data | [`to_vec`](Self::to_vec), [`into_dense_col_major_parts`](Self::into_dense_col_major_parts), [`sum`](Self::sum), [`only`](Self::only) |
/// | Contraction | [`contract`](Self::contract) |
/// | Arithmetic | [`add`](Self::add), [`scale`](Self::scale), [`axpby`](Self::axpby) |
/// | Factorization | via [`TensorFactorizationLike::factorize`](crate::TensorFactorizationLike::factorize) |
/// | Norms | [`norm`](Self::norm), [`norm_squared`](Self::norm_squared), [`maxabs`](Self::maxabs) |
/// | Index ops | [`replaceind`](Self::replaceind), [`permute_indices`](Self::permute_indices) |
/// # Data Layout
/// Logical dense extraction uses **column-major** order (first index varies
/// fastest), matching Fortran, Julia, and ITensors.jl conventions. Compact
/// structured payloads additionally carry explicit payload dimensions, strides,
/// and logical-axis classes.
/// # Examples
/// ```
/// use tensor4all_core::{TensorDynLen, DynIndex};
/// // Create a 2x3 real tensor
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let t = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
/// assert_eq!(t.dims(), vec![2, 3]);
/// assert!(t.is_f64());
/// // Sum all elements: 1+2+3+4+5+6 = 21
/// let s = t.sum().unwrap();
/// assert!((s.real() - 21.0).abs() < 1e-12);
/// // Extract data back out
/// let data_out = t.to_vec::<f64>().unwrap();
/// assert_eq!(data_out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
#[derive(Clone)]
pub struct TensorDynLen {
    /// Full index information (includes tags and other metadata).
    pub indices: Vec<DynIndex>,
    /// Authoritative payload representation. Compact storage is used when the
    /// dtype is supported by [`Storage`]; otherwise this retains an eager
    /// payload without promotion.
    pub(crate) storage: TensorDynLenStorage,
    /// Lazily materialized logical-dense eager payload for native execution and AD.
    pub(crate) eager_cache: Arc<OnceLock<Arc<EagerTensor>>>,
}

impl TensorDynLen {
    fn dense_axis_classes(rank: usize) -> Vec<usize> {
        (0..rank).collect()
    }

    fn dtype_name(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::C32 => "c32",
            DType::C64 => "c64",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::Bool => "bool",
        }
    }

    fn scalar_dtype(&self) -> Result<DType> {
        if let Some(inner) = self.storage.eager() {
            return Ok(inner.data().dtype());
        }
        if self.storage.is_f64() {
            Ok(DType::F64)
        } else if self.storage.is_c64() {
            Ok(DType::C64)
        } else {
            Err(anyhow::anyhow!(
                "unable to determine TensorDynLen scalar dtype"
            ))
        }
    }

    fn diag_axis_classes(rank: usize) -> Vec<usize> {
        if rank == 0 {
            vec![]
        } else {
            vec![0; rank]
        }
    }

    fn canonicalize_axis_classes(axis_classes: &[usize]) -> Vec<usize> {
        let mut map = std::collections::HashMap::new();
        let mut next = 0usize;
        axis_classes
            .iter()
            .map(|&class_id| {
                *map.entry(class_id).or_insert_with(|| {
                    let canonical = next;
                    next += 1;
                    canonical
                })
            })
            .collect()
    }

    fn permute_axis_classes(&self, perm: &[usize]) -> Vec<usize> {
        let axis_classes = self.storage.axis_classes();
        let permuted: Vec<usize> = perm.iter().map(|&index| axis_classes[index]).collect();
        Self::canonicalize_axis_classes(&permuted)
    }

    fn normalize_insert_axis(op: &str, axis: isize, rank: usize) -> Result<usize> {
        let normalized = if axis < 0 {
            rank as isize + 1 + axis
        } else {
            axis
        };
        anyhow::ensure!(
            normalized >= 0 && normalized <= rank as isize,
            "{op}: axis {axis} is out of bounds for inserting into rank {rank}"
        );
        Ok(normalized as usize)
    }

    fn is_diag_axis_classes(axis_classes: &[usize]) -> bool {
        axis_classes.len() >= 2 && axis_classes.iter().all(|&class_id| class_id == 0)
    }

    fn einsum_subscripts_from_usize_ids(
        inputs: &[Vec<usize>],
        output: &[usize],
    ) -> Result<EinsumSubscripts> {
        let input_labels = inputs
            .iter()
            .map(|ids| {
                ids.iter()
                    .map(|&id| {
                        u32::try_from(id)
                            .map_err(|_| anyhow::anyhow!("einsum label {id} exceeds u32 range"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let output_labels = output
            .iter()
            .map(|&id| {
                u32::try_from(id)
                    .map_err(|_| anyhow::anyhow!("einsum label {id} exceeds u32 range"))
            })
            .collect::<Result<Vec<_>>>()?;
        let input_refs = input_labels.iter().map(Vec::as_slice).collect::<Vec<_>>();
        Ok(EinsumSubscripts::new(&input_refs, &output_labels))
    }

    fn build_binary_einsum_subscripts(
        lhs_rank: usize,
        axes_a: &[usize],
        rhs_rank: usize,
        axes_b: &[usize],
    ) -> Result<EinsumSubscripts> {
        anyhow::ensure!(
            axes_a.len() == axes_b.len(),
            "contract axis length mismatch: lhs {:?}, rhs {:?}",
            axes_a,
            axes_b
        );

        let mut lhs_ids = vec![usize::MAX; lhs_rank];
        let mut rhs_ids = vec![usize::MAX; rhs_rank];
        let mut next_id = 0usize;

        let mut seen_lhs = vec![false; lhs_rank];
        let mut seen_rhs = vec![false; rhs_rank];

        for (&lhs_axis, &rhs_axis) in axes_a.iter().zip(axes_b.iter()) {
            anyhow::ensure!(
                lhs_axis < lhs_rank,
                "lhs contract axis {lhs_axis} out of range"
            );
            anyhow::ensure!(
                rhs_axis < rhs_rank,
                "rhs contract axis {rhs_axis} out of range"
            );
            anyhow::ensure!(
                !seen_lhs[lhs_axis],
                "duplicate lhs contract axis {lhs_axis}"
            );
            anyhow::ensure!(
                !seen_rhs[rhs_axis],
                "duplicate rhs contract axis {rhs_axis}"
            );
            seen_lhs[lhs_axis] = true;
            seen_rhs[rhs_axis] = true;
            lhs_ids[lhs_axis] = next_id;
            rhs_ids[rhs_axis] = next_id;
            next_id += 1;
        }

        let mut output_ids = Vec::with_capacity(lhs_rank + rhs_rank - 2 * axes_a.len());
        for id in &mut lhs_ids {
            if *id == usize::MAX {
                *id = next_id;
                output_ids.push(next_id);
                next_id += 1;
            }
        }
        for id in &mut rhs_ids {
            if *id == usize::MAX {
                *id = next_id;
                output_ids.push(next_id);
                next_id += 1;
            }
        }

        Self::einsum_subscripts_from_usize_ids(&[lhs_ids, rhs_ids], &output_ids)
    }

    fn binary_dot_general_config(axes_a: &[usize], axes_b: &[usize]) -> Result<DotGeneralConfig> {
        anyhow::ensure!(
            axes_a.len() == axes_b.len(),
            "contract axis length mismatch: lhs {:?}, rhs {:?}",
            axes_a,
            axes_b
        );
        Ok(DotGeneralConfig {
            lhs_contracting_dims: axes_a.to_vec(),
            rhs_contracting_dims: axes_b.to_vec(),
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        })
    }

    fn binary_contraction_axis_classes(
        lhs_axis_classes: &[usize],
        axes_a: &[usize],
        rhs_axis_classes: &[usize],
        axes_b: &[usize],
    ) -> Vec<usize> {
        debug_assert_eq!(axes_a.len(), axes_b.len());

        fn find(parent: &mut [usize], value: usize) -> usize {
            if parent[value] != value {
                parent[value] = find(parent, parent[value]);
            }
            parent[value]
        }

        fn union(parent: &mut [usize], lhs: usize, rhs: usize) {
            let lhs_root = find(parent, lhs);
            let rhs_root = find(parent, rhs);
            if lhs_root != rhs_root {
                parent[rhs_root] = lhs_root;
            }
        }

        let lhs_payload_rank = lhs_axis_classes
            .iter()
            .copied()
            .max()
            .map(|value| value + 1)
            .unwrap_or(0);
        let rhs_payload_rank = rhs_axis_classes
            .iter()
            .copied()
            .max()
            .map(|value| value + 1)
            .unwrap_or(0);
        let rhs_offset = lhs_payload_rank;
        let mut parent: Vec<usize> = (0..lhs_payload_rank + rhs_payload_rank).collect();

        for (&lhs_axis, &rhs_axis) in axes_a.iter().zip(axes_b.iter()) {
            union(
                &mut parent,
                lhs_axis_classes[lhs_axis],
                rhs_offset + rhs_axis_classes[rhs_axis],
            );
        }

        let mut lhs_contracted = vec![false; lhs_axis_classes.len()];
        for &axis in axes_a {
            lhs_contracted[axis] = true;
        }
        let mut rhs_contracted = vec![false; rhs_axis_classes.len()];
        for &axis in axes_b {
            rhs_contracted[axis] = true;
        }

        let mut root_to_class = std::collections::HashMap::new();
        let mut next_class = 0usize;
        let mut axis_classes = Vec::new();

        for (axis, &class_id) in lhs_axis_classes.iter().enumerate() {
            if !lhs_contracted[axis] {
                let root = find(&mut parent, class_id);
                let class = *root_to_class.entry(root).or_insert_with(|| {
                    let value = next_class;
                    next_class += 1;
                    value
                });
                axis_classes.push(class);
            }
        }
        for (axis, &class_id) in rhs_axis_classes.iter().enumerate() {
            if !rhs_contracted[axis] {
                let root = find(&mut parent, rhs_offset + class_id);
                let class = *root_to_class.entry(root).or_insert_with(|| {
                    let value = next_class;
                    next_class += 1;
                    value
                });
                axis_classes.push(class);
            }
        }

        axis_classes
    }

    fn scale_subscripts(rank: usize) -> Result<EinsumSubscripts> {
        let ids: Vec<usize> = (0..rank).collect();
        Self::einsum_subscripts_from_usize_ids(&[ids.clone(), Vec::new()], &ids)
    }

    fn scale_target_dtype(payload: DType, scalar: DType) -> Result<DType> {
        let target = match payload {
            DType::F32 => match scalar {
                DType::C32 | DType::C64 => DType::C32,
                DType::F32 | DType::F64 => DType::F32,
                dtype => {
                    return Err(anyhow::anyhow!(
                        "unsupported scalar dtype {dtype:?} for f32 scaling"
                    ));
                }
            },
            DType::C32 => match scalar {
                DType::F32 | DType::F64 | DType::C32 | DType::C64 => DType::C32,
                dtype => {
                    return Err(anyhow::anyhow!(
                        "unsupported scalar dtype {dtype:?} for c32 scaling"
                    ));
                }
            },
            DType::F64 => match scalar {
                DType::C32 | DType::C64 => DType::C64,
                DType::F32 | DType::F64 => DType::F64,
                dtype => {
                    return Err(anyhow::anyhow!(
                        "unsupported scalar dtype {dtype:?} for f64 scaling"
                    ));
                }
            },
            DType::C64 => match scalar {
                DType::F32 | DType::F64 | DType::C32 | DType::C64 => DType::C64,
                dtype => {
                    return Err(anyhow::anyhow!(
                        "unsupported scalar dtype {dtype:?} for c64 scaling"
                    ));
                }
            },
            dtype => {
                return Err(anyhow::anyhow!(
                    "unsupported tensor dtype {dtype:?} for scaling"
                ));
            }
        };
        Ok(target)
    }

    fn validate_indices(indices: &[DynIndex]) -> Result<()> {
        let mut seen = HashSet::new();
        for idx in indices {
            anyhow::ensure!(
                seen.insert(idx.clone()),
                "Tensor indices must all be unique"
            );
        }
        Ok(())
    }

    fn validate_diag_dims(dims: &[usize]) -> Result<()> {
        if !dims.is_empty() {
            let first_dim = dims[0];
            for (i, &dim) in dims.iter().enumerate() {
                anyhow::ensure!(
                    dim == first_dim,
                    "DiagTensor requires all indices to have the same dimension, but dims[{i}] = {dim} != dims[0] = {first_dim}"
                );
            }
        }
        Ok(())
    }

    fn seed_native_payload(storage: &Storage, dims: &[usize]) -> Result<NativeTensor> {
        storage_to_native_tensor(storage, dims)
    }

    fn empty_eager_cache() -> Arc<OnceLock<Arc<EagerTensor>>> {
        Arc::new(OnceLock::new())
    }

    fn eager_cache_with(inner: EagerTensor) -> Arc<OnceLock<Arc<EagerTensor>>> {
        let cache = Arc::new(OnceLock::new());
        let _ = cache.set(Arc::new(inner));
        cache
    }

    fn compact_payload_inner(&self) -> Result<EagerTensor> {
        self.ensure_storage_ready()?;
        if let Some(inner) = self.storage.eager() {
            return Ok(inner.clone());
        }
        Ok(EagerTensor::from_tensor_in(
            storage_payload_native(self.storage.materialize(self.indices.len())?.as_ref())?,
            default_eager_ctx()?,
        ))
    }

    fn dense_inner_from_payload(
        payload: &EagerTensor,
        axis_classes: &[usize],
        logical_dims: &[usize],
    ) -> Result<EagerTensor> {
        let payload_rank = axis_classes
            .iter()
            .copied()
            .max()
            .map(|class_id| class_id + 1)
            .unwrap_or(0);
        anyhow::ensure!(
            payload.data().shape().len() == payload_rank,
            "structured payload rank {} does not match axis classes {:?}",
            payload.data().shape().len(),
            axis_classes
        );
        anyhow::ensure!(
            logical_dims.len() == axis_classes.len(),
            "logical rank {} does not match axis class rank {}",
            logical_dims.len(),
            axis_classes.len()
        );

        if axis_classes == Self::dense_axis_classes(logical_dims.len()) {
            anyhow::ensure!(
                payload.data().shape() == logical_dims,
                "dense payload dims {:?} do not match logical dims {:?}",
                payload.data().shape(),
                logical_dims
            );
            return Ok(payload.clone());
        }

        let mut first_axis_by_class = vec![None; payload_rank];
        let mut dense = payload.clone();
        for (logical_axis, &class_id) in axis_classes.iter().enumerate() {
            let first_axis = match first_axis_by_class[class_id] {
                Some(first_axis) => first_axis,
                None => {
                    first_axis_by_class[class_id] = Some(logical_axis);
                    continue;
                }
            };
            dense = dense.embed_diag(first_axis, logical_axis)?;
        }
        anyhow::ensure!(
            dense.data().shape() == logical_dims,
            "expanded structured payload dims {:?} do not match logical dims {:?}",
            dense.data().shape(),
            logical_dims
        );
        Ok(dense)
    }

    fn tracked_compact_payload_value(&self) -> Option<&StructuredPayload> {
        self.storage
            .deferred_error()
            .is_none()
            .then_some(self.storage.compact_payload())
            .flatten()
            .filter(|value| value.payload.tracks_grad())
    }

    fn ensure_storage_ready(&self) -> Result<()> {
        if let Some(error) = self.storage.deferred_error() {
            return Err(anyhow::Error::new(error.clone()));
        }
        Ok(())
    }

    fn compact_payload_is_logical_dense(&self, payload_dims: &[usize]) -> bool {
        self.storage.axis_classes() == Self::dense_axis_classes(self.indices.len())
            && payload_dims == self.dims()
    }

    fn uses_tracked_compact_storage(&self) -> bool {
        self.tracked_compact_payload_value()
            .is_some_and(|value| !self.compact_payload_is_logical_dense(&value.payload_dims))
    }

    fn ensure_shape_packing_preserves_ad(&self, op_name: &str) -> Result<()> {
        self.ensure_storage_ready()?;
        anyhow::ensure!(
            !self.uses_tracked_compact_storage(),
            "{op_name}: structured AD tensors with compact storage are not supported because materializing compact storage would detach gradients"
        );
        Ok(())
    }

    fn operand_indices_for_contraction(&self, conjugate: bool) -> Vec<DynIndex> {
        if conjugate {
            self.indices.iter().map(|index| index.conj()).collect()
        } else {
            self.indices.clone()
        }
    }

    fn build_binary_contraction_labels(
        lhs_rank: usize,
        axes_a: &[usize],
        rhs_rank: usize,
        axes_b: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        anyhow::ensure!(
            axes_a.len() == axes_b.len(),
            "contract axis length mismatch: lhs {:?}, rhs {:?}",
            axes_a,
            axes_b
        );

        let mut lhs_ids = vec![usize::MAX; lhs_rank];
        let mut rhs_ids = vec![usize::MAX; rhs_rank];
        let mut next_id = 0usize;

        let mut seen_lhs = vec![false; lhs_rank];
        let mut seen_rhs = vec![false; rhs_rank];

        for (&lhs_axis, &rhs_axis) in axes_a.iter().zip(axes_b.iter()) {
            anyhow::ensure!(
                lhs_axis < lhs_rank,
                "lhs contract axis {lhs_axis} out of range"
            );
            anyhow::ensure!(
                rhs_axis < rhs_rank,
                "rhs contract axis {rhs_axis} out of range"
            );
            anyhow::ensure!(
                !seen_lhs[lhs_axis],
                "duplicate lhs contract axis {lhs_axis}"
            );
            anyhow::ensure!(
                !seen_rhs[rhs_axis],
                "duplicate rhs contract axis {rhs_axis}"
            );
            seen_lhs[lhs_axis] = true;
            seen_rhs[rhs_axis] = true;
            lhs_ids[lhs_axis] = next_id;
            rhs_ids[rhs_axis] = next_id;
            next_id += 1;
        }

        let mut output_ids = Vec::with_capacity(lhs_rank + rhs_rank - 2 * axes_a.len());
        for id in &mut lhs_ids {
            if *id == usize::MAX {
                *id = next_id;
                output_ids.push(next_id);
                next_id += 1;
            }
        }
        for id in &mut rhs_ids {
            if *id == usize::MAX {
                *id = next_id;
                output_ids.push(next_id);
                next_id += 1;
            }
        }

        Ok((lhs_ids, rhs_ids, output_ids))
    }

    fn build_payload_einsum_subscripts(
        input_roots: &[Vec<usize>],
        output_roots: &[usize],
    ) -> Result<EinsumSubscripts> {
        Self::einsum_subscripts_from_usize_ids(input_roots, output_roots)
    }

    fn normalize_eager_payload_for_roots(
        payload: &EagerTensor,
        roots: &[usize],
    ) -> Result<(Option<EagerTensor>, Vec<usize>)> {
        anyhow::ensure!(
            payload.data().shape().len() == roots.len(),
            "payload rank {} does not match root label count {}",
            payload.data().shape().len(),
            roots.len()
        );

        let mut current_payload = None;
        let mut current_roots = roots.to_vec();
        while let Some((axis_a, axis_b)) = Self::first_duplicate_pair(&current_roots) {
            let source = current_payload.as_ref().unwrap_or(payload);
            current_payload = Some(source.extract_diag(axis_a, axis_b)?);
            current_roots.remove(axis_b);
        }

        Ok((current_payload, current_roots))
    }

    fn first_duplicate_pair(values: &[usize]) -> Option<(usize, usize)> {
        let mut first_axis_by_value = std::collections::HashMap::new();
        for (axis, &value) in values.iter().enumerate() {
            if let Some(&first_axis) = first_axis_by_value.get(&value) {
                return Some((first_axis, axis));
            }
            first_axis_by_value.insert(value, axis);
        }
        None
    }

    fn from_structured_payload_inner(
        indices: Vec<DynIndex>,
        payload_inner: EagerTensor,
        payload_dims: Vec<usize>,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        Self::validate_indices(&indices)?;
        if payload_inner.data().shape() != payload_dims {
            return Err(anyhow::anyhow!(
                "structured payload dims {:?} do not match planned payload dims {:?}",
                payload_inner.data().shape(),
                payload_dims
            ));
        }
        if axis_classes == Self::dense_axis_classes(indices.len()) {
            return Self::from_inner_with_axis_classes(indices, payload_inner, axis_classes);
        }
        let structured_payload = Arc::new(StructuredPayload {
            payload: Arc::new(payload_inner),
            payload_dims,
            axis_classes,
        });
        Ok(Self {
            indices,
            storage: TensorDynLenStorage::Compact(structured_payload),
            eager_cache: Self::empty_eager_cache(),
        })
    }

    fn contract_structured_payloads(
        &self,
        other: &Self,
        result_indices: Vec<DynIndex>,
        axes_a: &[usize],
        axes_b: &[usize],
    ) -> Result<Self> {
        let (lhs_labels, rhs_labels, output_labels) = Self::build_binary_contraction_labels(
            self.indices.len(),
            axes_a,
            other.indices.len(),
            axes_b,
        )?;
        Self::contract_structured_payloads_nary(
            &[self, other],
            result_indices,
            vec![lhs_labels, rhs_labels],
            output_labels,
        )
    }

    pub(crate) fn contract_structured_payloads_nary(
        operands: &[&Self],
        result_indices: Vec<DynIndex>,
        input_labels: Vec<Vec<usize>>,
        output_labels: Vec<usize>,
    ) -> Result<Self> {
        anyhow::ensure!(
            !operands.is_empty(),
            "structured contraction needs operands"
        );
        for operand in operands {
            operand.ensure_storage_ready()?;
        }
        let layouts = operands
            .iter()
            .map(|operand| {
                OperandLayout::new(operand.dims(), operand.storage.axis_classes().to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let spec = StructuredContractionSpec {
            input_labels,
            output_labels,
            retained_labels: Default::default(),
        };
        let plan = StructuredContractionPlan::new(&layouts, &spec)?;
        let any_grad = operands.iter().any(|operand| operand.tracks_grad());

        if any_grad {
            let dtypes = operands
                .iter()
                .map(|operand| {
                    operand.storage.dtype().ok_or_else(|| {
                        anyhow::anyhow!("structured contraction operand has no scalar dtype")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let target = Self::common_eager_dtype(&dtypes)?;
            let mut payloads = Vec::with_capacity(operands.len());
            for operand in operands {
                let payload = operand.compact_payload_inner()?;
                payloads.push(if payload.data().dtype() == target {
                    payload
                } else {
                    payload.convert(target)?
                });
            }

            let mut normalized = Vec::with_capacity(payloads.len());
            let mut labels = Vec::with_capacity(payloads.len());
            for (operand_idx, (payload, operand_plan)) in
                payloads.iter().zip(plan.operand_plans.iter()).enumerate()
            {
                let (payload, roots) =
                    Self::normalize_eager_payload_for_roots(payload, &operand_plan.class_roots)?;
                normalized.push(payload.unwrap_or_else(|| payloads[operand_idx].clone()));
                labels.push(roots);
            }
            let refs = normalized.iter().collect::<Vec<_>>();
            let subscripts =
                Self::build_payload_einsum_subscripts(&labels, &plan.output_payload_roots)?;
            let payload = eager_einsum_ad(&refs, &subscripts)?;
            return Self::from_structured_payload_inner(
                result_indices,
                payload,
                plan.output_payload_dims,
                plan.output_axis_classes,
            );
        }

        // The native backend borrows contiguous compact payloads and promotes
        // only operands whose compact dtype differs. No logical dense tensor is
        // constructed on this path.
        let storage_owners = operands
            .iter()
            .map(|operand| match &operand.storage {
                TensorDynLenStorage::Materialized(storage) => Some(Arc::clone(storage)),
                TensorDynLenStorage::Deferred { source, .. } => match source.as_ref() {
                    TensorDynLenStorage::Materialized(storage) => Some(Arc::clone(storage)),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut inputs = Vec::with_capacity(operands.len());
        for (operand_idx, operand) in operands.iter().enumerate() {
            if let Some(storage) = storage_owners[operand_idx].as_ref() {
                inputs.push(storage_payload_native_read_input(storage.as_ref())?);
            } else {
                let inner = operand
                    .storage
                    .eager()
                    .ok_or_else(|| anyhow::anyhow!("structured operand has no compact payload"))?;
                inputs.push(NativeTensorReadInput::Borrowed(
                    tenferro::TensorRead::from_tensor(inner.data()),
                ));
            }
        }
        let mut normalized = Vec::with_capacity(inputs.len());
        let mut labels = Vec::with_capacity(inputs.len());
        for (input, operand_plan) in inputs.into_iter().zip(plan.operand_plans.iter()) {
            let (input, roots) =
                normalize_payload_read_for_roots(input, &operand_plan.class_roots)?;
            normalized.push(input);
            labels.push(roots);
        }
        let refs = normalized
            .iter()
            .zip(labels.iter())
            .map(|(input, labels)| (input, labels.as_slice()))
            .collect::<Vec<_>>();
        let payload = tensor4all_tensorbackend::einsum_native_tensor_reads(
            &refs,
            &plan.output_payload_roots,
        )?;
        let payload_inner = EagerTensor::from_tensor_in(payload, default_eager_ctx()?);
        Self::from_structured_payload_inner(
            result_indices,
            payload_inner,
            plan.output_payload_dims,
            plan.output_axis_classes,
        )
    }

    fn common_eager_dtype(dtypes: &[DType]) -> Result<DType> {
        let target = if dtypes.contains(&DType::C64)
            || (dtypes.contains(&DType::C32) && dtypes.contains(&DType::F64))
        {
            DType::C64
        } else if dtypes.contains(&DType::F64) || dtypes.contains(&DType::C32) {
            if dtypes.contains(&DType::C32) {
                DType::C32
            } else {
                DType::F64
            }
        } else {
            DType::F32
        };
        anyhow::ensure!(
            dtypes.iter().all(|dtype| {
                matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64)
            }),
            "structured contraction supports only f32, f64, c32, and c64 operands"
        );
        Ok(target)
    }

    fn should_use_structured_payload_contract(&self, other: &Self) -> bool {
        self.tracks_grad()
            || other.tracks_grad()
            || self.storage.axis_classes() != Self::dense_axis_classes(self.indices.len())
            || other.storage.axis_classes() != Self::dense_axis_classes(other.indices.len())
    }

    fn storage_from_native_with_axis_classes(
        native: &NativeTensor,
        axis_classes: &[usize],
        logical_rank: usize,
    ) -> Result<Storage> {
        if matches!(native.dtype(), DType::F32 | DType::C32) {
            return Err(anyhow::anyhow!(
                "compact TensorDynLen storage does not support dtype {:?}; retain the eager payload",
                native.dtype()
            ));
        }
        if Self::is_diag_axis_classes(axis_classes) {
            match native.dtype() {
                DType::F64 | DType::I32 | DType::I64 | DType::Bool => Storage::from_diag_col_major(
                    native_tensor_primal_to_diag_f64(native)?,
                    logical_rank,
                ),
                DType::C64 => Storage::from_diag_col_major(
                    native_tensor_primal_to_diag_c64(native)?,
                    logical_rank,
                ),
                DType::F32 | DType::C32 => Err(anyhow::anyhow!(
                    "compact TensorDynLen storage does not support dtype {:?}",
                    native.dtype()
                )),
            }
        } else {
            storage_from_payload_native(native.clone(), native.shape(), axis_classes.to_vec())
        }
    }

    fn dense_selected_diag_payload<T: TensorElement + Copy + Zero>(
        payload: Vec<T>,
        kept_dims: &[usize],
        selected_positions: &[usize],
    ) -> Vec<T> {
        let output_len = kept_dims.iter().product::<usize>();
        let mut data = vec![T::zero(); output_len];
        if output_len == 0 {
            return data;
        }

        let Some((&first_position, rest)) = selected_positions.split_first() else {
            return data;
        };
        if rest.iter().any(|&position| position != first_position) {
            return data;
        }

        let value = payload[first_position];
        if kept_dims.is_empty() {
            data[0] = value;
            return data;
        }

        let mut offset = 0usize;
        let mut stride = 1usize;
        for &dim in kept_dims {
            offset += first_position * stride;
            stride *= dim;
        }
        data[offset] = value;
        data
    }

    fn select_diag_indices(
        &self,
        kept_indices: Vec<DynIndex>,
        kept_dims: Vec<usize>,
        positions: &[usize],
    ) -> Result<Self> {
        if self.storage.is_f64() {
            let storage = self.storage.materialize(self.indices.len())?;
            let payload = storage
                .payload_f64_col_major_vec()
                .map_err(anyhow::Error::new)?;
            let data = Self::dense_selected_diag_payload(payload, &kept_dims, positions);
            Self::from_dense(kept_indices, data)
        } else if self.storage.is_c64() {
            let storage = self.storage.materialize(self.indices.len())?;
            let payload = storage
                .payload_c64_col_major_vec()
                .map_err(anyhow::Error::new)?;
            let data = Self::dense_selected_diag_payload(payload, &kept_dims, positions);
            Self::from_dense(kept_indices, data)
        } else if self.storage.dtype() == Some(DType::F32) {
            let payload = self
                .storage
                .eager()
                .and_then(|inner| inner.data().as_slice::<f32>())
                .ok_or_else(|| anyhow::anyhow!("failed to read f32 diagonal payload"))?
                .to_vec();
            let data = Self::dense_selected_diag_payload(payload, &kept_dims, positions);
            Self::from_dense(kept_indices, data)
        } else if self.storage.dtype() == Some(DType::C32) {
            let payload = self
                .storage
                .eager()
                .and_then(|inner| inner.data().as_slice::<Complex32>())
                .ok_or_else(|| anyhow::anyhow!("failed to read c32 diagonal payload"))?
                .to_vec();
            let data = Self::dense_selected_diag_payload(payload, &kept_dims, positions);
            Self::from_dense(kept_indices, data)
        } else {
            Err(anyhow::anyhow!("unsupported diagonal storage scalar type"))
        }
    }

    fn col_major_strides(dims: &[usize]) -> Result<Vec<isize>> {
        let mut strides = Vec::with_capacity(dims.len());
        let mut stride = 1isize;
        for &dim in dims {
            strides.push(stride);
            let dim = isize::try_from(dim)
                .map_err(|_| anyhow::anyhow!("dimension does not fit in isize"))?;
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("column-major stride overflow"))?;
        }
        Ok(strides)
    }

    fn zero_structured_selection<T>(
        kept_indices: Vec<DynIndex>,
        kept_dims: &[usize],
    ) -> Result<Self>
    where
        T: TensorElement + Zero,
    {
        let output_len = checked_product(kept_dims)?;
        Self::from_dense(kept_indices, vec![T::zero(); output_len])
    }

    fn selected_structured_class_positions(
        axis_classes: &[usize],
        payload_rank: usize,
        selected_axes: &[usize],
        positions: &[usize],
    ) -> Option<Vec<Option<usize>>> {
        let mut selected_class_positions = vec![None; payload_rank];
        for (&axis, &position) in selected_axes.iter().zip(positions.iter()) {
            let class_id = axis_classes[axis];
            if let Some(existing) = selected_class_positions[class_id] {
                if existing != position {
                    return None;
                }
            } else {
                selected_class_positions[class_id] = Some(position);
            }
        }
        Some(selected_class_positions)
    }

    fn select_structured_indices_typed<T, F>(
        &self,
        payload: Vec<T>,
        kept_axes: &[usize],
        kept_indices: Vec<DynIndex>,
        kept_dims: Vec<usize>,
        selected: (&[usize], &[usize]),
        make_output: F,
    ) -> Result<Self>
    where
        T: TensorElement + Zero,
        F: FnOnce(Vec<T>, Vec<usize>, Vec<isize>, Vec<usize>) -> Result<Self>,
    {
        let (selected_axes, positions) = selected;
        let payload_dims = self.storage.payload_dims();
        let axis_classes = self.storage.axis_classes();
        let payload_rank = payload_dims.len();
        let Some(selected_class_positions) = Self::selected_structured_class_positions(
            axis_classes,
            payload_rank,
            selected_axes,
            positions,
        ) else {
            return Self::zero_structured_selection::<T>(kept_indices, &kept_dims);
        };

        let selected_class_kept = kept_axes
            .iter()
            .any(|&axis| selected_class_positions[axis_classes[axis]].is_some());
        if selected_class_kept {
            return self.select_structured_indices_dense(
                payload,
                kept_axes,
                kept_indices,
                kept_dims,
                &selected_class_positions,
            );
        }

        let mut old_to_new_class = vec![None; payload_rank];
        let mut output_payload_dims = Vec::new();
        let mut output_axis_classes = Vec::with_capacity(kept_axes.len());
        for &axis in kept_axes {
            let class_id = axis_classes[axis];
            let new_class = match old_to_new_class[class_id] {
                Some(new_class) => new_class,
                None => {
                    let new_class = output_payload_dims.len();
                    old_to_new_class[class_id] = Some(new_class);
                    output_payload_dims.push(payload_dims[class_id]);
                    new_class
                }
            };
            output_axis_classes.push(new_class);
        }

        let output_len = checked_product(&output_payload_dims)?;
        let mut output_payload = Vec::with_capacity(output_len);
        for linear in 0..output_len {
            let output_payload_index = decode_col_major_linear(linear, &output_payload_dims)?;
            let mut input_payload_index = vec![0usize; payload_rank];
            for class_id in 0..payload_rank {
                input_payload_index[class_id] =
                    if let Some(position) = selected_class_positions[class_id] {
                        position
                    } else if let Some(new_class) = old_to_new_class[class_id] {
                        output_payload_index[new_class]
                    } else {
                        return Err(anyhow::anyhow!(
                            "structured payload class {class_id} is neither selected nor kept"
                        ));
                    };
            }
            let input_linear = encode_col_major_linear(&input_payload_index, payload_dims)?;
            output_payload.push(payload[input_linear]);
        }

        let output_strides = Self::col_major_strides(&output_payload_dims)?;
        make_output(
            output_payload,
            output_payload_dims,
            output_strides,
            output_axis_classes,
        )
    }

    fn select_structured_indices_dense<T>(
        &self,
        payload: Vec<T>,
        kept_axes: &[usize],
        kept_indices: Vec<DynIndex>,
        kept_dims: Vec<usize>,
        selected_class_positions: &[Option<usize>],
    ) -> Result<Self>
    where
        T: TensorElement + Zero,
    {
        let payload_dims = self.storage.payload_dims();
        let axis_classes = self.storage.axis_classes();
        let output_len = checked_product(&kept_dims)?;
        let mut output = Vec::with_capacity(output_len);

        for linear in 0..output_len {
            let kept_position = decode_col_major_linear(linear, &kept_dims)?;
            let mut input_payload_index = selected_class_positions.to_vec();
            let mut is_structural_zero = false;

            for (&axis, &position) in kept_axes.iter().zip(kept_position.iter()) {
                let class_id = axis_classes[axis];
                match input_payload_index[class_id] {
                    Some(existing) if existing != position => {
                        is_structural_zero = true;
                        break;
                    }
                    Some(_) => {}
                    None => input_payload_index[class_id] = Some(position),
                }
            }

            if is_structural_zero {
                output.push(T::zero());
                continue;
            }

            let input_payload_index = input_payload_index
                .into_iter()
                .enumerate()
                .map(|(class_id, position)| {
                    position.ok_or_else(|| {
                        anyhow::anyhow!(
                            "structured payload class {class_id} is neither selected nor kept"
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let input_linear = encode_col_major_linear(&input_payload_index, payload_dims)?;
            output.push(payload[input_linear]);
        }

        Self::from_dense(kept_indices, output)
    }

    fn select_structured_indices(
        &self,
        kept_axes: &[usize],
        kept_indices: Vec<DynIndex>,
        kept_dims: Vec<usize>,
        selected_axes: &[usize],
        positions: &[usize],
    ) -> Result<Self> {
        if self.storage.is_f64() {
            let storage = self.storage.materialize(self.indices.len())?;
            let payload = storage
                .payload_f64_col_major_vec()
                .map_err(anyhow::Error::new)?;
            let output_indices = kept_indices.clone();
            self.select_structured_indices_typed(
                payload,
                kept_axes,
                kept_indices,
                kept_dims,
                (selected_axes, positions),
                move |payload, dims, strides, classes| {
                    let storage = Storage::new_structured(payload, dims, strides, classes)?;
                    Self::from_storage(output_indices, Arc::new(storage))
                },
            )
        } else if self.storage.is_c64() {
            let storage = self.storage.materialize(self.indices.len())?;
            let payload = storage
                .payload_c64_col_major_vec()
                .map_err(anyhow::Error::new)?;
            let output_indices = kept_indices.clone();
            self.select_structured_indices_typed(
                payload,
                kept_axes,
                kept_indices,
                kept_dims,
                (selected_axes, positions),
                move |payload, dims, strides, classes| {
                    let storage = Storage::new_structured(payload, dims, strides, classes)?;
                    Self::from_storage(output_indices, Arc::new(storage))
                },
            )
        } else if self.storage.dtype() == Some(DType::F32) {
            let payload = self
                .storage
                .eager()
                .and_then(|inner| inner.data().as_slice::<f32>())
                .ok_or_else(|| anyhow::anyhow!("failed to read f32 structured payload"))?
                .to_vec();
            let output_indices = kept_indices.clone();
            self.select_structured_indices_typed(
                payload,
                kept_axes,
                kept_indices,
                kept_dims,
                (selected_axes, positions),
                move |payload, dims, _strides, classes| {
                    let native = dense_native_tensor_from_col_major(&payload, &dims)?;
                    let inner = EagerTensor::from_tensor_in(native, default_eager_ctx()?);
                    Self::from_structured_payload_inner(output_indices, inner, dims, classes)
                },
            )
        } else if self.storage.dtype() == Some(DType::C32) {
            let payload = self
                .storage
                .eager()
                .and_then(|inner| inner.data().as_slice::<Complex32>())
                .ok_or_else(|| anyhow::anyhow!("failed to read c32 structured payload"))?
                .to_vec();
            let output_indices = kept_indices.clone();
            self.select_structured_indices_typed(
                payload,
                kept_axes,
                kept_indices,
                kept_dims,
                (selected_axes, positions),
                move |payload, dims, _strides, classes| {
                    let native = dense_native_tensor_from_col_major(&payload, &dims)?;
                    let inner = EagerTensor::from_tensor_in(native, default_eager_ctx()?);
                    Self::from_structured_payload_inner(output_indices, inner, dims, classes)
                },
            )
        } else {
            Err(anyhow::anyhow!(
                "unsupported structured storage scalar type"
            ))
        }
    }

    fn validate_storage_matches_indices(indices: &[DynIndex], storage: &Storage) -> Result<()> {
        let dims = Self::expected_dims_from_indices(indices);
        let storage_dims = storage.logical_dims();
        if storage_dims != dims {
            return Err(anyhow::anyhow!(
                "storage logical dims {:?} do not match indices dims {:?}",
                storage_dims,
                dims
            ));
        }
        if storage.is_diag() {
            Self::validate_diag_dims(&dims)?;
        }
        Ok(())
    }

    fn try_materialized_inner(&self) -> Result<&EagerTensor> {
        self.ensure_storage_ready()?;
        let logical_dims = self.dims();
        if let Some(value) = self.tracked_compact_payload_value() {
            if self.compact_payload_is_logical_dense(&value.payload_dims) {
                return Ok(value.payload.as_ref());
            }
            if self.eager_cache.get().is_none() {
                let dense = Self::dense_inner_from_payload(
                    value.payload.as_ref(),
                    &value.axis_classes,
                    &logical_dims,
                )?;
                let _ = self.eager_cache.set(Arc::new(dense));
            }
            return self
                .eager_cache
                .get()
                .map(|inner| inner.as_ref())
                .ok_or_else(|| {
                    anyhow::anyhow!("TensorDynLen structured AD cache was not initialized")
                });
        }
        if let Some(inner) = self.storage.eager() {
            if self.storage.axis_classes() == Self::dense_axis_classes(self.indices.len()) {
                return Ok(inner);
            }
            if self.eager_cache.get().is_none() {
                let dense = Self::dense_inner_from_payload(
                    inner,
                    self.storage.axis_classes(),
                    &logical_dims,
                )?;
                let _ = self.eager_cache.set(Arc::new(dense));
            }
            return self
                .eager_cache
                .get()
                .map(|inner| inner.as_ref())
                .ok_or_else(|| {
                    anyhow::anyhow!("TensorDynLen structured eager cache was not initialized")
                });
        }
        if self.eager_cache.get().is_none() {
            let native = profile_pairwise_contract_section("materialize_storage_to_native", || {
                let storage = self.storage.materialize(self.indices.len())?;
                Self::seed_native_payload(storage.as_ref(), &logical_dims)
            })
            .context("TensorDynLen materialization failed")?;
            record_pairwise_contract_profile_bytes(
                "materialize_storage_to_native",
                native_tensor_profile_bytes(&native),
            );
            let _ = self.eager_cache.set(Arc::new(EagerTensor::from_tensor_in(
                native,
                default_eager_ctx()?,
            )));
        }
        self.eager_cache
            .get()
            .map(|inner| inner.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!("TensorDynLen materialization cache was not initialized")
            })
    }

    pub(crate) fn as_inner(&self) -> Result<&EagerTensor> {
        self.try_materialized_inner()
    }

    /// Compute dims from `indices` order.
    #[inline]
    fn expected_dims_from_indices(indices: &[DynIndex]) -> Vec<usize> {
        indices.iter().map(|idx| idx.dim()).collect()
    }

    /// Get dims in the current `indices` order.
    ///
    /// This is computed on-demand from `indices` (single source of truth).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(3);
    /// let k = DynIndex::new_dyn(4);
    /// let t = TensorDynLen::from_dense(
    ///     vec![i, j, k],
    ///     vec![0.0; 24],
    /// ).unwrap();
    /// assert_eq!(t.dims(), vec![2, 3, 4]);
    /// ```
    pub fn dims(&self) -> Vec<usize> {
        Self::expected_dims_from_indices(&self.indices)
    }

    /// Select fixed coordinates for tensor indices and drop those axes.
    ///
    /// The `selected_indices` slice identifies tensor axes by index identity,
    /// and `positions` gives the zero-based coordinate to take on each
    /// selected axis. Unselected indices are preserved in their original order.
    ///
    /// # Arguments
    ///
    /// * `selected_indices` - Indices to fix and remove from the result. Each
    ///   index must appear exactly once in this tensor.
    /// * `positions` - Coordinates for `selected_indices`. Each coordinate must
    ///   be less than the corresponding index dimension.
    ///
    /// # Returns
    ///
    /// A tensor over the unselected indices. Selecting no indices returns a
    /// clone of the original tensor. Selecting all indices returns a rank-0
    /// scalar tensor. Diagonal and structured tensors are sliced from their
    /// compact payload without materializing the original full tensor; the
    /// result keeps structured storage when the remaining logical axes can
    /// still be represented by axis classes.
    ///
    /// # Errors
    /// Returns an error when a selected coordinate is out of range for its index
    /// (an out-of-bounds failure) or when `selected_indices` and `positions`
    /// differ in length (a length mismatch).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
    ///
    /// let selected = tensor.select_indices(&[j], &[1]).unwrap();
    /// assert_eq!(selected.dims(), vec![2]);
    /// assert_eq!(selected.to_vec::<f64>().unwrap(), vec![3.0, 4.0]);
    /// ```
    pub fn select_indices(
        &self,
        selected_indices: &[DynIndex],
        positions: &[usize],
    ) -> Result<Self> {
        if selected_indices.len() != positions.len() {
            return Err(anyhow::anyhow!(
                "selected_indices length {} does not match positions length {}",
                selected_indices.len(),
                positions.len()
            ));
        }
        if selected_indices.is_empty() {
            return Ok(self.clone());
        }

        let mut selected_axes = Vec::with_capacity(selected_indices.len());
        let mut seen_axes = HashSet::with_capacity(selected_indices.len());
        for (selected, &position) in selected_indices.iter().zip(positions.iter()) {
            let axis = self
                .indices
                .iter()
                .position(|index| index == selected)
                .ok_or_else(|| anyhow::anyhow!("selected index is not present in tensor"))?;
            if !seen_axes.insert(axis) {
                return Err(anyhow::anyhow!("selected index appears more than once"));
            }
            let dim = self.indices[axis].dim();
            if position >= dim {
                return Err(anyhow::anyhow!(
                    "selected coordinate {position} is out of range for axis {axis} with dim {dim}"
                ));
            }
            selected_axes.push(axis);
        }

        let kept_axes = self
            .indices
            .iter()
            .enumerate()
            .filter(|(axis, _)| !seen_axes.contains(axis))
            .map(|(axis, _)| axis)
            .collect::<Vec<_>>();
        let kept_indices = kept_axes
            .iter()
            .map(|&axis| self.indices[axis].clone())
            .collect::<Vec<_>>();
        let kept_dims = kept_axes
            .iter()
            .map(|&axis| self.indices[axis].dim())
            .collect::<Vec<_>>();

        if self.storage.storage_kind() == StorageKind::Diagonal {
            return self.select_diag_indices(kept_indices, kept_dims, positions);
        }
        if self.storage.storage_kind() == StorageKind::Structured {
            return self.select_structured_indices(
                &kept_axes,
                kept_indices,
                kept_dims,
                &selected_axes,
                positions,
            );
        }
        if self.storage.storage_kind() != StorageKind::Dense {
            return Err(anyhow::anyhow!(
                "select_indices got unsupported storage kind {:?}",
                self.storage.storage_kind()
            ));
        }

        let rank = self.indices.len();
        let mut starts = vec![0_i64; rank];
        let mut slice_sizes = self.dims();
        for (&axis, &position) in selected_axes.iter().zip(positions.iter()) {
            starts[axis] = i64::try_from(position)
                .map_err(|_| anyhow::anyhow!("selected coordinate does not fit in i64"))?;
            slice_sizes[axis] = 1;
        }

        let starts_tensor = EagerTensor::from_tensor_in(
            NativeTensor::from_vec_col_major(vec![rank], starts),
            default_eager_ctx()?,
        );
        let sliced = self
            .try_materialized_inner()?
            .dynamic_slice(&starts_tensor, &slice_sizes)?;
        Self::from_inner(kept_indices, sliced.reshape(&kept_dims)?)
    }

    /// Stack tensors along a newly inserted index.
    ///
    /// Each input must have exactly the same index order and dimensions. The
    /// `new_index` dimension must match the number of input tensors. The
    /// `axis` argument follows tenferro/PyTorch-style insertion semantics:
    /// `0` inserts before the first existing axis and `-1` appends a trailing
    /// axis. Use `axis = -1` for batched contractions because tenferro uses
    /// trailing batch dimensions as the canonical batched-GEMM layout.
    ///
    /// # Errors
    ///
    /// Returns an error if no tensors are provided, the new index dimension
    /// does not match the number of tensors, an input has a different index
    /// order, `axis` is outside the valid insertion range, or a tracked
    /// structured-AD tensor uses compact storage that would need dense
    /// materialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let batch = DynIndex::new_dyn(2);
    /// let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap();
    /// let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap();
    ///
    /// let stacked = TensorDynLen::stack_along_new_index(&[&a, &b], batch.clone(), -1).unwrap();
    ///
    /// assert_eq!(stacked.indices(), &[i, batch]);
    /// assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn stack_along_new_index(
        tensors: &[&Self],
        new_index: DynIndex,
        axis: isize,
    ) -> Result<Self> {
        let first = tensors
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("stack_along_new_index requires at least one tensor"))?;
        anyhow::ensure!(
            new_index.dim() == tensors.len(),
            "stack_along_new_index: new index dim {} does not match tensor count {}",
            new_index.dim(),
            tensors.len()
        );

        let base_indices = first.indices.clone();
        for tensor in tensors.iter().copied().skip(1) {
            anyhow::ensure!(
                tensor.indices == base_indices,
                "stack_along_new_index: input tensors must have identical index order"
            );
        }
        for &tensor in tensors {
            tensor.ensure_shape_packing_preserves_ad("stack_along_new_index")?;
        }

        let insert_axis =
            Self::normalize_insert_axis("stack_along_new_index", axis, base_indices.len())?;
        let mut result_indices = base_indices;
        result_indices.insert(insert_axis, new_index);

        let inner_refs = tensors
            .iter()
            .map(|tensor| tensor.try_materialized_inner())
            .collect::<Result<Vec<_>>>()?;
        let stacked = EagerTensor::stack(&inner_refs, axis)?;
        Self::from_inner(result_indices, stacked)
    }

    /// Select positions along one index and replace it with a new index.
    ///
    /// This is the retained-axis counterpart to [`Self::select_indices`]:
    /// instead of fixing one coordinate and removing the index, it gathers a
    /// list of positions and keeps the gathered axis under `target_index`.
    /// Repeated positions are allowed; reverse-mode AD accumulates repeated
    /// cotangents through tenferro's scatter-add gather transpose.
    ///
    /// # Errors
    /// Returns an error when a selected position is out of range for the source
    /// index (an out-of-bounds failure) or when the source and target index
    /// dimensions are incompatible (a dimension mismatch).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let source = DynIndex::new_dyn(3);
    /// let target = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(
    ///     vec![source.clone()],
    ///     vec![10.0_f64, 20.0, 30.0],
    /// ).unwrap();
    ///
    /// let selected = tensor.index_select(&source, target.clone(), &[2, 0]).unwrap();
    ///
    /// assert_eq!(selected.indices(), &[target]);
    /// assert_eq!(selected.to_vec::<f64>().unwrap(), vec![30.0, 10.0]);
    /// ```
    pub fn index_select(
        &self,
        source_index: &DynIndex,
        target_index: DynIndex,
        positions: &[usize],
    ) -> Result<Self> {
        anyhow::ensure!(
            target_index.dim() == positions.len(),
            "index_select: target index dim {} does not match position count {}",
            target_index.dim(),
            positions.len()
        );
        let axis = self
            .indices
            .iter()
            .position(|index| index == source_index)
            .ok_or_else(|| anyhow::anyhow!("index_select: source index is not present"))?;
        let source_dim = self.indices[axis].dim();
        for &position in positions {
            anyhow::ensure!(
                position < source_dim,
                "index_select: position {position} is out of range for source dim {source_dim}"
            );
        }
        self.ensure_shape_packing_preserves_ad("index_select")?;

        let axis = isize::try_from(axis)
            .map_err(|_| anyhow::anyhow!("index_select: axis does not fit in isize"))?;
        let selected = self
            .try_materialized_inner()?
            .index_select(axis, positions)?;
        let mut result_indices = self.indices.clone();
        result_indices[axis as usize] = target_index;
        Self::from_inner(result_indices, selected)
    }

    /// Create a new tensor with dynamic rank.
    ///
    /// # Errors
    /// Returns an error when the storage logical dimension does not match the
    /// index dimension product (a dimension mismatch) or when duplicate
    /// indices are provided.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::Storage;
    /// use std::sync::Arc;
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let storage = Arc::new(Storage::new_dense::<f64>(3).unwrap());
    /// let t = TensorDynLen::new(vec![i], storage).unwrap();
    /// assert_eq!(t.dims(), vec![3]);
    /// ```
    pub fn new(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
        Self::from_storage(indices, storage)
    }

    /// Create a new tensor with dynamic rank, automatically computing dimensions from indices.
    ///
    /// This is a convenience constructor that extracts dimensions from indices using `IndexLike::dim()`.
    ///
    /// # Errors
    /// Returns an error when the storage logical dimension does not match the
    /// index dimension product (a dimension mismatch) or when duplicate
    /// indices are provided.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::Storage;
    /// use std::sync::Arc;
    ///
    /// let i = DynIndex::new_dyn(4);
    /// let storage = Arc::new(Storage::new_dense::<f64>(4).unwrap());
    /// let t = TensorDynLen::from_indices(vec![i], storage).unwrap();
    /// assert_eq!(t.dims(), vec![4]);
    /// ```
    pub fn from_indices(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
        Self::new(indices, storage)
    }

    /// Create a tensor from explicit compact storage.
    ///
    /// # Errors
    /// Returns an error when the storage scalar kind is incompatible with the
    /// requested operations (a scalar-kind mismatch) or the storage cannot
    /// represent the given index space (a dimension mismatch).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::Storage;
    /// use std::sync::Arc;
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let storage = Arc::new(Storage::new_diag(vec![1.0_f64, 2.0]).unwrap());
    /// let t = TensorDynLen::from_storage(vec![i, j], storage).unwrap();
    /// assert_eq!(t.dims(), vec![2, 2]);
    /// ```
    pub fn from_storage(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
        Self::validate_indices(&indices)?;
        Self::validate_storage_matches_indices(&indices, storage.as_ref())?;
        Ok(Self {
            indices,
            storage: TensorDynLenStorage::from_storage(storage),
            eager_cache: Self::empty_eager_cache(),
        })
    }

    /// Create a tensor from explicit structured storage.
    ///
    /// This is an alias for [`TensorDynLen::from_storage`] with a name that
    /// emphasizes that compact structured metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the structured storage is invalid (an invalid-storage
    /// failure) or the index space is incompatible (a dimension mismatch).
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::{Storage, StorageKind};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let storage = Arc::new(Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap());
    /// let tensor = TensorDynLen::from_structured_storage(vec![i, j], storage).unwrap();
    /// assert_eq!(tensor.storage().unwrap().storage_kind(), StorageKind::Diagonal);
    /// ```
    pub fn from_structured_storage(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
        Self::from_storage(indices, storage)
    }

    /// Construct a compact copy tensor that selects one physical-site value.
    ///
    /// The returned rank-3 tensor has logical indices `[left, site, right]` and
    /// value `scale` exactly when `left == right` and `site == selected_value`;
    /// every other entry is zero. Its payload has `left.dim * site.dim`
    /// elements rather than `left.dim * site.dim * right.dim` dense elements.
    ///
    /// # Arguments
    ///
    /// - `left`: left copy axis; its dimension must be positive and equal to
    ///   `right.dim`.
    /// - `site`: physical axis whose selected coordinate remains active.
    /// - `right`: right copy axis paired with `left`.
    /// - `selected_value`: zero-based coordinate in `0..site.dim`.
    /// - `scale`: value stored on the selected copy diagonal.
    ///
    /// # Returns
    ///
    /// A structured tensor with axis classes `[0, 1, 0]`. For `f64` and
    /// `Complex64`, compact storage is retained; `f32` and `Complex32` keep
    /// an eager authoritative payload because compact storage has no 32-bit
    /// scalar representation.
    ///
    /// # Errors
    ///
    /// Returns [`StructuredSelectorError`] when dimensions are zero or
    /// inconsistent, the selected value is out of bounds, checked size or
    /// stride arithmetic overflows, allocation fails, or backend structured
    /// storage validation rejects the metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::StorageKind;
    ///
    /// let left = DynIndex::new_dyn(2);
    /// let site = DynIndex::new_dyn(3);
    /// let right = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_copy_selector(
    ///     left,
    ///     site,
    ///     right,
    ///     1,
    ///     2.5_f64,
    /// ).unwrap();
    ///
    /// assert_eq!(tensor.storage().unwrap().storage_kind(), StorageKind::Structured);
    /// assert_eq!(tensor.storage().unwrap().payload_len(), 6);
    /// assert_eq!(
    ///     tensor.to_vec::<f64>().unwrap(),
    ///     vec![0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0],
    /// );
    /// ```
    pub fn from_copy_selector<T>(
        left: DynIndex,
        site: DynIndex,
        right: DynIndex,
        selected_value: usize,
        scale: T,
    ) -> std::result::Result<Self, StructuredSelectorError>
    where
        T: TensorElement + Copy + Zero,
    {
        if left.dim == 0 {
            return Err(StructuredSelectorError::ZeroDimension { axis: "left" });
        }
        if site.dim == 0 {
            return Err(StructuredSelectorError::ZeroDimension { axis: "site" });
        }
        if right.dim == 0 {
            return Err(StructuredSelectorError::ZeroDimension { axis: "right" });
        }
        if left.dim != right.dim {
            return Err(StructuredSelectorError::BondDimensionMismatch {
                left: left.dim,
                right: right.dim,
            });
        }
        if selected_value >= site.dim {
            return Err(StructuredSelectorError::SelectedValueOutOfBounds {
                value: selected_value,
                site_dim: site.dim,
            });
        }

        let payload_len =
            left.dim
                .checked_mul(site.dim)
                .ok_or(StructuredSelectorError::PayloadSizeOverflow {
                    bond_dim: left.dim,
                    site_dim: site.dim,
                })?;
        let _site_stride = isize::try_from(left.dim)
            .map_err(|_| StructuredSelectorError::StrideOverflow { bond_dim: left.dim })?;
        let selected_offset = left.dim.checked_mul(selected_value).ok_or(
            StructuredSelectorError::PayloadSizeOverflow {
                bond_dim: left.dim,
                site_dim: site.dim,
            },
        )?;

        let mut payload = Vec::new();
        payload.try_reserve_exact(payload_len).map_err(|_| {
            StructuredSelectorError::AllocationFailed {
                elements: payload_len,
            }
        })?;
        payload.resize(payload_len, T::zero());
        for bond in 0..left.dim {
            payload[selected_offset + bond] = scale;
        }
        let payload_native = dense_native_tensor_from_col_major(&payload, &[left.dim, site.dim])
            .map_err(|error| StructuredSelectorError::InvalidStorage {
                message: error.to_string(),
            })?;
        let payload_inner = EagerTensor::from_tensor_in(
            payload_native.clone(),
            default_eager_ctx().map_err(|error| StructuredSelectorError::InvalidStorage {
                message: error.to_string(),
            })?,
        );
        let payload_dims = vec![left.dim, site.dim];
        let indices = vec![left, site, right];
        if !matches!(
            payload_native.dtype(),
            DType::F32 | DType::F64 | DType::C32 | DType::C64
        ) {
            return Err(StructuredSelectorError::InvalidStorage {
                message: format!("unsupported selector dtype {:?}", payload_native.dtype()),
            });
        }
        Self::from_structured_payload_inner(indices, payload_inner, payload_dims, vec![0, 1, 0])
            .map_err(|error| StructuredSelectorError::InvalidStorage {
                message: error.to_string(),
            })
    }

    /// Create a tensor from a native tenferro payload.
    pub(crate) fn from_native(indices: Vec<DynIndex>, native: NativeTensor) -> Result<Self> {
        let axis_classes = Self::dense_axis_classes(indices.len());
        Self::from_native_with_axis_classes(indices, native, axis_classes)
    }

    pub(crate) fn from_native_with_axis_classes(
        indices: Vec<DynIndex>,
        native: NativeTensor,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        Self::from_inner_with_axis_classes(
            indices,
            EagerTensor::from_tensor_in(native, default_eager_ctx()?),
            axis_classes,
        )
    }

    pub(crate) fn from_inner(indices: Vec<DynIndex>, inner: EagerTensor) -> Result<Self> {
        let axis_classes = Self::dense_axis_classes(indices.len());
        Self::from_inner_with_axis_classes(indices, inner, axis_classes)
    }

    /// Compute the Hermitian eigendecomposition of a rank-2 tensor.
    ///
    /// The tensor must have two square matrix axes. The returned eigenvectors
    /// stay in [`TensorDynLen`] form so downstream tensor algebra can preserve
    /// AD metadata where the backend supports it. Eigenvalues are returned as
    /// detached real primal values because truncation and rank selection are
    /// nonsmooth control-flow decisions.
    ///
    /// `hermitian_tol` controls the allowed imaginary part of complex
    /// eigenvalues after the backend solve; use a small non-negative value such
    /// as `1e-12` for numerically Hermitian inputs.
    ///
    /// # Errors
    /// Returns an error when the tensor is not rank-2, when the two indices have
    /// unequal dimensions (a shape or dimension mismatch), or when the
    /// eigensolver fails to converge (a non-convergence failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, TensorContractionLike, TensorDynLen};
    ///
    /// let row = DynIndex::new_dyn(2);
    /// let col = DynIndex::new_dyn(2);
    /// let matrix = TensorDynLen::from_dense(
    ///     vec![row.clone(), col.clone()],
    ///     vec![3.0_f64, 0.0, 0.0, 5.0],
    /// ).unwrap();
    ///
    /// let decomp = matrix.hermitian_eigendecomposition(1.0e-12).unwrap();
    /// let eigenvector = decomp
    ///     .eigenvectors
    ///     .select_indices(&[decomp.eigenvector_index.clone()], &[0])
    ///     .unwrap();
    /// let eigenvector_as_col = eigenvector.replaceind(&row, &col).unwrap();
    /// let applied = TensorDynLen::contract(&[&matrix, &eigenvector_as_col]).unwrap();
    /// let expected = eigenvector.scale(AnyScalar::new_real(decomp.eigenvalues[0])).unwrap();
    ///
    /// assert!(applied.isapprox(&expected, 1.0e-12, 0.0).unwrap());
    /// ```
    pub fn hermitian_eigendecomposition(
        &self,
        hermitian_tol: f64,
    ) -> Result<TensorHermitianEigendecomposition> {
        anyhow::ensure!(
            self.indices.len() == 2,
            "TensorDynLen::hermitian_eigendecomposition requires a rank-2 tensor, got rank {}",
            self.indices.len()
        );
        let dims = self.dims();
        anyhow::ensure!(
            dims[0] == dims[1],
            "TensorDynLen::hermitian_eigendecomposition requires a square matrix, got {}x{}",
            dims[0],
            dims[1]
        );
        anyhow::ensure!(
            dims[0] > 0,
            "TensorDynLen::hermitian_eigendecomposition requires a non-empty matrix"
        );
        anyhow::ensure!(
            hermitian_tol.is_finite() && hermitian_tol >= 0.0,
            "TensorDynLen::hermitian_eigendecomposition requires a finite non-negative tolerance"
        );

        let input = self.try_materialized_inner()?;
        let (values, vectors) = tenferro_linalg::eager_tensor::eigh(input)
            .map_err(|source| anyhow::anyhow!("Hermitian eigendecomposition failed: {source}"))?;

        let eigenvalue_index = DynIndex::new_dyn(dims[0]);
        let eigenvector_index = DynIndex::new_dyn(dims[0]);
        let eigenvalue_tensor = Self::from_inner(vec![eigenvalue_index], values)?;
        let eigenvalues = Self::read_real_eigenvalues(&eigenvalue_tensor, hermitian_tol)
            .with_context(|| {
                "TensorDynLen::hermitian_eigendecomposition failed to read eigenvalues"
            })?;
        let eigenvectors = Self::from_inner(
            vec![self.indices[0].clone(), eigenvector_index.clone()],
            vectors,
        )?;

        Ok(TensorHermitianEigendecomposition {
            eigenvalues,
            eigenvectors,
            eigenvector_index,
        })
    }

    fn read_real_eigenvalues(values: &Self, hermitian_tol: f64) -> Result<Vec<f64>> {
        if values.is_complex() {
            values
                .to_vec::<Complex64>()?
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    let imaginary = value.im.abs();
                    let allowed = hermitian_tol * value.norm().max(1.0);
                    anyhow::ensure!(
                        imaginary <= allowed,
                        "Hermitian eigenvalue {index} has imaginary part {imaginary}, exceeding tolerance {allowed}"
                    );
                    Ok(value.re)
                })
                .collect()
        } else {
            values.to_vec::<f64>()
        }
    }

    pub(crate) fn from_diag_inner(
        indices: Vec<DynIndex>,
        payload_inner: EagerTensor,
    ) -> Result<Self> {
        let dims = Self::expected_dims_from_indices(&indices);
        Self::validate_indices(&indices)?;
        Self::validate_diag_dims(&dims)?;
        Self::validate_diag_payload_len(payload_inner.data().shape().iter().product(), &dims)?;
        let axis_classes = Self::diag_axis_classes(dims.len());
        let diag_inner = payload_inner.embed_diag(0, 1)?;
        Self::from_inner_with_axis_classes(indices, diag_inner, axis_classes)
    }

    fn compact_inner_from_logical(
        inner: &EagerTensor,
        axis_classes: &[usize],
    ) -> Result<EagerTensor> {
        let mut payload = inner.clone();
        let mut classes = axis_classes.to_vec();
        while let Some((axis_a, axis_b)) = Self::first_duplicate_pair(&classes) {
            payload = payload.extract_diag(axis_a, axis_b)?;
            classes.remove(axis_b);
        }
        Ok(payload)
    }

    pub(crate) fn from_inner_with_axis_classes(
        indices: Vec<DynIndex>,
        inner: EagerTensor,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        let dims = profile_pairwise_contract_section("from_inner_expected_dims", || {
            Self::expected_dims_from_indices(&indices)
        });
        profile_pairwise_contract_section("from_inner_validate_indices", || {
            Self::validate_indices(&indices)
        })?;
        if dims != inner.data().shape() {
            return Err(anyhow::anyhow!(
                "native payload dims {:?} do not match indices dims {:?}",
                inner.data().shape(),
                dims
            ));
        }
        if Self::is_diag_axis_classes(&axis_classes) {
            profile_pairwise_contract_section("from_inner_validate_diag_dims", || {
                Self::validate_diag_dims(&dims)
            })?;
        }
        let storage = if axis_classes == Self::dense_axis_classes(indices.len()) {
            TensorDynLenStorage::from_eager_dense(inner, indices.len())
        } else {
            let payload = Self::compact_inner_from_logical(&inner, &axis_classes)?;
            let payload_dims = payload.data().shape().to_vec();
            TensorDynLenStorage::Compact(Arc::new(StructuredPayload {
                payload: Arc::new(payload),
                payload_dims,
                axis_classes,
            }))
        };
        Ok(Self {
            indices,
            storage,
            eager_cache: Self::empty_eager_cache(),
        })
    }

    /// Borrow the indices.
    pub fn indices(&self) -> &[DynIndex] {
        &self.indices
    }

    pub(crate) fn axis_classes(&self) -> &[usize] {
        self.storage.axis_classes()
    }

    /// Borrow the native payload.
    pub(crate) fn as_native(&self) -> Result<&NativeTensor> {
        Ok(self.try_materialized_inner()?.data())
    }

    /// Enable reverse-mode AD tracking on this tensor by creating a tracked leaf.
    /// # Errors
    /// Returns an error when the tensor is not a scalar (a rank mismatch) or the
    /// AD backend cannot track the tensor's dtype.
    ///
    pub fn enable_grad(self) -> Result<Self> {
        self.ensure_storage_ready()?;
        // Keep the eager payload when available: compact Storage currently
        // stores only f64/C64 and must not promote f32/C32 leaves before AD.
        let eager_payload = self
            .storage
            .eager()
            .or_else(|| self.eager_cache.get().map(AsRef::as_ref))
            .filter(|inner| inner.data().shape() == self.storage.payload_dims());
        let payload = match eager_payload {
            Some(inner) => inner.data().clone(),
            None => {
                let materialized = self.storage.materialize(self.indices.len())?;
                storage_payload_native(materialized.as_ref())
                    .context("TensorDynLen::enable_grad failed")?
            }
        };
        let payload_dims = self.storage.payload_dims().to_vec();
        let axis_classes = self.storage.axis_classes().to_vec();
        let tracked = Arc::new(EagerTensor::requires_grad_in(payload, default_eager_ctx()?));
        let storage = if axis_classes == Self::dense_axis_classes(self.indices.len()) {
            TensorDynLenStorage::Eager {
                inner: tracked,
                axis_classes,
            }
        } else {
            TensorDynLenStorage::Compact(Arc::new(StructuredPayload {
                payload: tracked,
                payload_dims,
                axis_classes,
            }))
        };
        Ok(Self {
            indices: self.indices,
            storage,
            eager_cache: Self::empty_eager_cache(),
        })
    }

    /// Report whether this tensor participates in gradient tracking.
    pub fn tracks_grad(&self) -> bool {
        self.storage.eager().is_some_and(EagerTensor::tracks_grad)
            || self
                .eager_cache
                .get()
                .is_some_and(|inner| inner.tracks_grad())
    }

    /// Return the accumulated gradient, if one has been stored.
    /// # Errors
    /// Returns an error when the tensor is not a tracked leaf or the gradient is
    /// unavailable for the tensor's dtype (an unavailable-gradient failure).
    ///
    pub fn grad(&self) -> Result<Option<Self>> {
        if let Some(value) = self.tracked_compact_payload_value() {
            return value
                .payload
                .grad()
                .map(|grad| {
                    if self.compact_payload_is_logical_dense(&value.payload_dims) {
                        return Self::from_native_with_axis_classes(
                            self.indices.clone(),
                            grad.as_ref().clone(),
                            value.axis_classes.clone(),
                        );
                    }
                    anyhow::ensure!(
                        grad.as_ref().shape() == value.payload_dims,
                        "gradient payload dims {:?} do not match {:?}",
                        grad.as_ref().shape(),
                        value.payload_dims
                    );
                    let gradient =
                        EagerTensor::from_tensor_in(grad.as_ref().clone(), default_eager_ctx()?);
                    Self::from_structured_payload_inner(
                        self.indices.clone(),
                        gradient,
                        value.payload_dims.clone(),
                        value.axis_classes.clone(),
                    )
                })
                .transpose();
        }
        self.try_materialized_inner()?
            .grad()
            .map(|grad| {
                Self::from_native_with_axis_classes(
                    self.indices.clone(),
                    grad.as_ref().clone(),
                    self.storage.axis_classes().to_vec(),
                )
            })
            .transpose()
    }

    /// Clear the accumulated gradient stored for this tensor.
    /// # Errors
    /// Returns an error when the tensor is not a tracked leaf (a missing-graph
    /// failure).
    ///
    pub fn clear_grad(&self) -> Result<()> {
        self.ensure_storage_ready()?;
        if let Some(value) = self.tracked_compact_payload_value() {
            value.payload.clear_grad();
        }
        if let Some(inner) = self.storage.eager() {
            inner.clear_grad();
        }
        if let Some(inner) = self.eager_cache.get() {
            inner.clear_grad();
        }
        Ok(())
    }

    /// Run reverse-mode autodiff from this scalar tensor.
    /// # Errors
    /// Returns an error when the tensor is not a scalar (a rank mismatch) or the
    /// reverse pass fails (a graph failure).
    ///
    pub fn backward(&self) -> Result<()> {
        if let Some(value) = self.tracked_compact_payload_value() {
            return value
                .payload
                .backward()
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("TensorDynLen::backward failed: {e}"));
        }
        self.try_materialized_inner()?
            .backward()
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("TensorDynLen::backward failed: {e}"))
    }

    /// Detach this tensor from the reverse graph.
    /// # Errors
    /// Returns an error when the tensor is not a tracked leaf (a missing-graph
    /// failure).
    ///
    pub fn detach(&self) -> Result<Self> {
        Self::from_inner_with_axis_classes(
            self.indices.clone(),
            self.try_materialized_inner()?.detach(),
            self.storage.axis_classes().to_vec(),
        )
    }

    /// Check if this tensor is already in canonical form.
    pub fn is_simple(&self) -> bool {
        true
    }

    /// Materialize the primal payload as a compact storage snapshot.
    ///
    /// The eager payload remains authoritative for `f32`/`c32` and tracked
    /// structured tensors; this method is a fallible bridge to compact storage.
    ///
    /// # Errors
    ///
    /// Returns [`TensorStorageError`] when an eager backend payload cannot be
    /// converted to compact storage, when its dtype is `f32`/`c32`, or when a
    /// deferred eager operation failed.
    pub fn to_storage(&self) -> std::result::Result<Arc<Storage>, TensorStorageError> {
        self.storage.materialize(self.indices.len())
    }

    /// Materializes and returns a compact storage snapshot.
    ///
    /// # Errors
    /// Returns an error when the compact storage cannot be materialized (a
    /// backend failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::StorageKind;
    ///
    /// let tensor = TensorDynLen::from_dense(
    ///     vec![DynIndex::new_dyn(2)],
    ///     vec![1.0_f64, 2.0],
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.storage().unwrap().storage_kind(), StorageKind::Dense);
    /// ```
    pub fn storage(&self) -> std::result::Result<Arc<Storage>, TensorStorageError> {
        self.storage.materialize(self.indices.len())
    }

    /// Return the logical storage layout without materializing compact storage.
    ///
    /// For `f32` and `c32`, the eager representation is authoritative because
    /// compact [`Storage`] supports only `f64` and `c64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::StorageKind;
    ///
    /// let tensor = TensorDynLen::from_diag(
    ///     vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
    ///     vec![1.0_f32, 2.0],
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.storage_kind(), StorageKind::Diagonal);
    /// ```
    pub fn storage_kind(&self) -> StorageKind {
        self.storage.storage_kind()
    }

    /// Sum all elements, returning `AnyScalar`.
    ///
    /// # Errors
    /// Returns an error when the reduction fails (a backend or scalar-extraction
    /// failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let t = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0, 3.0]).unwrap();
    /// let s = t.sum().unwrap();
    /// assert!((s.real() - 6.0).abs() < 1e-12);
    /// ```
    pub fn sum(&self) -> Result<AnyScalar> {
        self.ensure_storage_ready()?;
        if self.indices.is_empty() {
            return AnyScalar::from_tensor(self.clone());
        }
        if let Some(payload) = self.storage.eager().filter(|payload| payload.tracks_grad()) {
            let axes: Vec<usize> = (0..payload.data().shape().len()).collect();
            let reduced = payload.reduce_sum(&axes)?;
            return AnyScalar::from_tensor(Self::from_inner(Vec::new(), reduced)?);
        }
        self.storage.sum_scalar()
    }

    /// Extract the scalar value from a 0-dimensional tensor (or 1-element tensor).
    ///
    /// This is similar to Julia's `only()` function.
    ///
    /// # Errors
    /// Returns an error when the tensor is not rank-0 and does not contain exactly
    /// one element (a rank mismatch).
    /// # Panics
    ///
    /// Panics if the tensor has more than one element.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor4all_core::{TensorDynLen, AnyScalar};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// // Create a scalar tensor (0 dimensions, 1 element)
    /// let indices: Vec<Index<DynId>> = vec![];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![42.0]).unwrap();
    ///
    /// assert_eq!(tensor.only().unwrap().real(), 42.0);
    /// ```
    pub fn only(&self) -> Result<AnyScalar> {
        let dims = self.dims();
        let total_size = checked_product(&dims)?;
        anyhow::ensure!(
            total_size == 1 || dims.is_empty(),
            "only() requires a scalar tensor (1 element), got {} elements with dims {:?}",
            if dims.is_empty() { 1 } else { total_size },
            dims
        );
        self.sum()
    }

    /// Permute the tensor dimensions using the given new indices order.
    ///
    /// This is the main permutation method that takes the desired new indices
    /// and automatically computes the corresponding permutation of dimensions
    /// and data. The new indices must be a permutation of the original indices
    /// (matched by ID).
    ///
    /// # Arguments
    /// * `new_indices` - The desired new indices order. Must be a permutation
    ///   of `self.indices` (matched by ID).
    ///
    /// # Errors
    /// Returns an error when `new_order` does not contain exactly the tensor's
    /// indices (an index-set mismatch or a missing-index failure).
    /// # Panics
    /// Panics if `new_indices.len() != self.indices.len()`, if any index ID
    /// doesn't match, or if there are duplicate indices.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// // Create a 2×3 tensor
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let indices = vec![i.clone(), j.clone()];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Permute to 3×2: swap the two dimensions by providing new indices order
    /// let permuted = tensor.permute_indices(&[j, i]).unwrap();
    /// assert_eq!(permuted.dims(), vec![3, 2]);
    /// ```
    pub fn permute_indices(&self, new_indices: &[DynIndex]) -> Result<Self> {
        // Compute permutation by matching IDs
        let perm = compute_permutation_from_indices(&self.indices, new_indices)?;
        if perm.iter().copied().eq(0..perm.len()) {
            return Ok(Self {
                indices: new_indices.to_vec(),
                storage: self.storage.clone(),
                eager_cache: Arc::clone(&self.eager_cache),
            });
        }

        let permuted = self.try_materialized_inner()?.transpose(&perm)?;
        let axis_classes = self.permute_axis_classes(&perm);
        Self::from_inner_with_axis_classes(new_indices.to_vec(), permuted, axis_classes)
    }

    /// Permute the tensor dimensions, returning a new tensor.
    ///
    /// This method reorders the indices, dimensions, and data according to the
    /// given permutation. The permutation specifies which old axis each new
    /// axis corresponds to: `new_axis[i] = old_axis[perm[i]]`.
    ///
    /// # Arguments
    /// * `perm` - The permutation: `perm[i]` is the old axis index for new axis `i`
    ///
    /// # Errors
    /// Returns an error when `new_order` does not contain exactly the tensor's
    /// indices (an index-set mismatch or a missing-index failure).
    /// # Panics
    /// Panics if `perm.len() != self.indices.len()` or if the permutation is invalid.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// // Create a 2×3 tensor
    /// let indices = vec![
    ///     Index::new_dyn(2),
    ///     Index::new_dyn(3),
    /// ];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Permute to 3×2: swap the two dimensions
    /// let permuted = tensor.permute(&[1, 0]).unwrap();
    /// assert_eq!(permuted.dims(), vec![3, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            perm.len() == self.indices.len(),
            "permutation length must match tensor rank"
        );
        let mut seen = HashSet::new();
        for &axis in perm {
            anyhow::ensure!(
                axis < self.indices.len(),
                "permutation axis {axis} out of range"
            );
            anyhow::ensure!(seen.insert(axis), "duplicate axis {axis} in permutation");
        }
        if perm.iter().copied().eq(0..perm.len()) {
            return Ok(self.clone());
        }

        // Permute indices
        let new_indices: Vec<DynIndex> = perm.iter().map(|&i| self.indices[i].clone()).collect();
        let permuted = self.try_materialized_inner()?.transpose(perm)?;
        let axis_classes = self.permute_axis_classes(perm);
        Self::from_inner_with_axis_classes(new_indices, permuted, axis_classes)
    }

    pub(crate) fn try_contract_pairwise_default(&self, other: &Self) -> Result<Self> {
        self.try_contract_pairwise_default_with_options(other, PairwiseContractionOptions::new())
    }

    pub(crate) fn try_contract_pairwise_default_with_options(
        &self,
        other: &Self,
        options: PairwiseContractionOptions,
    ) -> Result<Self> {
        let self_indices = profile_pairwise_contract_section("operand_indices", || {
            self.operand_indices_for_contraction(options.lhs_conj)
        });
        let other_indices = profile_pairwise_contract_section("operand_indices", || {
            other.operand_indices_for_contraction(options.rhs_conj)
        });
        let self_dims = profile_pairwise_contract_section("expected_dims", || {
            Self::expected_dims_from_indices(&self_indices)
        });
        let other_dims = profile_pairwise_contract_section("expected_dims", || {
            Self::expected_dims_from_indices(&other_indices)
        });
        let spec = profile_pairwise_contract_section("prepare_contraction", || {
            prepare_contraction(&self_indices, &self_dims, &other_indices, &other_dims)
        })
        .context("contraction preparation failed")?;
        let result_axis_classes = profile_pairwise_contract_section("result_axis_classes", || {
            Self::binary_contraction_axis_classes(
                self.storage.axis_classes(),
                &spec.axes_a,
                other.storage.axis_classes(),
                &spec.axes_b,
            )
        });

        if profile_pairwise_contract_section("structured_check", || {
            self.should_use_structured_payload_contract(other)
        }) {
            if options.has_conj() {
                let lhs = if options.lhs_conj {
                    self.conj()
                } else {
                    self.clone()
                };
                let rhs = if options.rhs_conj {
                    other.conj()
                } else {
                    other.clone()
                };
                return profile_pairwise_contract_section("structured_conj_fallback", || {
                    lhs.try_contract_pairwise_default(&rhs)
                });
            }
            return profile_pairwise_contract_section("structured_payload_contract", || {
                self.contract_structured_payloads(
                    other,
                    spec.result_indices.into_vec(),
                    &spec.axes_a,
                    &spec.axes_b,
                )
            });
        }

        if self.indices.is_empty() && other.indices.is_empty() {
            if options.has_conj() {
                let lhs = if options.lhs_conj {
                    self.conj()
                } else {
                    self.clone()
                };
                let rhs = if options.rhs_conj {
                    other.conj()
                } else {
                    other.clone()
                };
                return lhs.try_contract_pairwise_default(&rhs);
            }
            let result = profile_pairwise_contract_section("scalar_mul", || {
                Ok::<_, anyhow::Error>(
                    self.try_materialized_inner()?
                        .mul(other.try_materialized_inner()?)?,
                )
            })?;
            return profile_pairwise_contract_section("from_inner", || {
                Self::from_inner(spec.result_indices.into_vec(), result)
            });
        }

        let self_native = profile_pairwise_contract_section("as_native", || self.as_native())?;
        let other_native = profile_pairwise_contract_section("as_native", || other.as_native())?;
        if self_native.dtype() != other_native.dtype() {
            if options.has_conj() {
                let lhs = if options.lhs_conj {
                    self.conj()
                } else {
                    self.clone()
                };
                let rhs = if options.rhs_conj {
                    other.conj()
                } else {
                    other.clone()
                };
                return lhs.try_contract_pairwise_default(&rhs);
            }
            let result_native = profile_pairwise_contract_section("native_contract", || {
                contract_native_tensor(self_native, &spec.axes_a, other_native, &spec.axes_b)
            })?;
            return profile_pairwise_contract_section("from_native", || {
                Self::from_native_with_axis_classes(
                    spec.result_indices.into_vec(),
                    result_native,
                    result_axis_classes,
                )
            });
        }

        let config = profile_pairwise_contract_section("build_dot_general_config", || {
            Self::binary_dot_general_config(&spec.axes_a, &spec.axes_b)
        })?;
        let result = profile_pairwise_contract_section("dot_general_with_conj", || {
            let lhs = profile_pairwise_contract_section("lhs_try_materialized_inner", || {
                self.try_materialized_inner()
            })?;
            let rhs = profile_pairwise_contract_section("rhs_try_materialized_inner", || {
                other.try_materialized_inner()
            })?;
            profile_pairwise_contract_section("dot_general_execute", || {
                lhs.dot_general_with_conj(rhs, &config, options.lhs_conj, options.rhs_conj)
            })
            .map_err(anyhow::Error::from)
        })?;
        record_pairwise_contract_profile_bytes(
            "dot_general_output",
            native_tensor_profile_bytes(result.data()),
        );
        profile_pairwise_contract_section("from_inner_axis_classes", || {
            Self::from_inner_with_axis_classes(
                spec.result_indices.into_vec(),
                result,
                result_axis_classes,
            )
        })
    }

    pub(crate) fn try_tensordot_pairwise_explicit(
        &self,
        other: &Self,
        pairs: &[(DynIndex, DynIndex)],
    ) -> Result<Self> {
        use crate::index_ops::ContractionError;

        let self_dims = Self::expected_dims_from_indices(&self.indices);
        let other_dims = Self::expected_dims_from_indices(&other.indices);
        let spec = prepare_contraction_pairs(
            &self.indices,
            &self_dims,
            &other.indices,
            &other_dims,
            pairs,
        )
        .map_err(|e| match e {
            ContractionError::NoCommonIndices => {
                anyhow::anyhow!("tensordot: No pairs specified for contraction")
            }
            ContractionError::BatchContractionNotImplemented => anyhow::anyhow!(
                "tensordot: Common index found but not in contraction pairs. \
                         Batch contraction is not yet implemented."
            ),
            ContractionError::IndexNotFound { tensor } => {
                anyhow::anyhow!("tensordot: Index not found in {} tensor", tensor)
            }
            ContractionError::DimensionMismatch {
                pos_a,
                pos_b,
                dim_a,
                dim_b,
            } => anyhow::anyhow!(
                "tensordot: Dimension mismatch: self[{}]={} != other[{}]={}",
                pos_a,
                dim_a,
                pos_b,
                dim_b
            ),
            ContractionError::DuplicateAxis { tensor, pos } => {
                anyhow::anyhow!("tensordot: Duplicate axis {} in {} tensor", pos, tensor)
            }
        })?;
        let result_axis_classes = Self::binary_contraction_axis_classes(
            self.storage.axis_classes(),
            &spec.axes_a,
            other.storage.axis_classes(),
            &spec.axes_b,
        );

        if self.should_use_structured_payload_contract(other) {
            return self.contract_structured_payloads(
                other,
                spec.result_indices.into_vec(),
                &spec.axes_a,
                &spec.axes_b,
            );
        }

        if self.indices.is_empty() && other.indices.is_empty() {
            let result = self
                .try_materialized_inner()?
                .mul(other.try_materialized_inner()?)
                .map_err(|e| anyhow::anyhow!("tensordot scalar multiply failed: {e}"))?;
            return Self::from_inner(spec.result_indices.into_vec(), result);
        }

        let self_native = self.as_native()?;
        let other_native = other.as_native()?;
        if self_native.dtype() != other_native.dtype() {
            let result_native =
                contract_native_tensor(self_native, &spec.axes_a, other_native, &spec.axes_b)?;
            return Self::from_native_with_axis_classes(
                spec.result_indices.into_vec(),
                result_native,
                result_axis_classes,
            );
        }

        let subscripts = Self::build_binary_einsum_subscripts(
            self.indices.len(),
            &spec.axes_a,
            other.indices.len(),
            &spec.axes_b,
        )?;
        let result = eager_einsum_ad(
            &[
                self.try_materialized_inner()?,
                other.try_materialized_inner()?,
            ],
            &subscripts,
        )
        .map_err(|e| anyhow::anyhow!("tensordot failed: {e}"))?;
        Self::from_inner_with_axis_classes(
            spec.result_indices.into_vec(),
            result,
            result_axis_classes,
        )
    }

    pub(crate) fn try_outer_product_pairwise(&self, other: &Self) -> Result<Self> {
        use anyhow::Context;

        // Check for common indices - outer product should have none
        let common_positions = common_ind_positions(&self.indices, &other.indices);
        if !common_positions.is_empty() {
            let common_ids: Vec<_> = common_positions
                .iter()
                .map(|(pos_a, _)| self.indices[*pos_a].id())
                .collect();
            return Err(anyhow::anyhow!(
                "outer_product: tensors have common indices {:?}. \
                 Use tensordot to contract common indices, or use sim() to replace \
                 indices with fresh IDs before computing outer product.",
                common_ids
            ))
            .context("outer_product: common indices found");
        }

        // Build result indices and dimensions
        let mut result_indices = self.indices.clone();
        result_indices.extend(other.indices.iter().cloned());
        let result_axis_classes = Self::binary_contraction_axis_classes(
            self.storage.axis_classes(),
            &[],
            other.storage.axis_classes(),
            &[],
        );
        if self.should_use_structured_payload_contract(other) {
            return self.contract_structured_payloads(other, result_indices, &[], &[]);
        }
        let self_native = self.as_native()?;
        let other_native = other.as_native()?;
        if self_native.dtype() != other_native.dtype() {
            let result_native = contract_native_tensor(self_native, &[], other_native, &[])?;
            return Self::from_native_with_axis_classes(
                result_indices,
                result_native,
                result_axis_classes,
            );
        }

        let subscripts = Self::build_binary_einsum_subscripts(
            self.indices.len(),
            &[],
            other.indices.len(),
            &[],
        )?;
        let result = eager_einsum_ad(
            &[
                self.try_materialized_inner()?,
                other.try_materialized_inner()?,
            ],
            &subscripts,
        )
        .map_err(|e| anyhow::anyhow!("outer_product failed: {e}"))?;
        Self::from_inner_with_axis_classes(result_indices, result, result_axis_classes)
    }
}

// ============================================================================
// Random tensor generation
// ============================================================================

impl TensorDynLen {
    /// Create a random tensor with values from standard normal distribution (generic over scalar type).
    ///
    /// For `f64`, each element is drawn from the standard normal distribution.
    /// For `Complex64`, both real and imaginary parts are drawn independently.
    ///
    /// # Type Parameters
    /// * `T` - The scalar element type (must implement [`RandomScalar`])
    /// * `R` - The random number generator type
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `indices` - The indices for the tensor
    ///
    /// # Errors
    /// Returns an error when the dimension product overflows (an overflow failure)
    /// or the backend cannot generate the requested scalar type.
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    ///
    /// let mut rng = ChaCha8Rng::seed_from_u64(42);
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![i, j]).unwrap();
    /// assert_eq!(tensor.dims(), vec![2, 3]);
    /// ```
    pub fn random<T: RandomScalar, R: Rng>(rng: &mut R, indices: Vec<DynIndex>) -> Result<Self> {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
        let size = checked_product(&dims)?;
        let data: Vec<T> = (0..size).map(|_| T::random_value(rng)).collect();
        Self::from_dense(indices, data)
    }
}

impl TensorDynLen {
    /// Add two tensors element-wise.
    ///
    /// The tensors must have the same index set (matched by ID). If the indices
    /// are in a different order, the other tensor will be permuted to match `self`.
    ///
    /// # Arguments
    /// * `other` - The tensor to add
    ///
    /// # Returns
    /// A new tensor representing `self + other`, or an error if:
    /// - The tensors have different index sets
    /// - The dimensions don't match
    /// - Storage types are incompatible
    ///
    /// # Errors
    /// Returns an error when the two tensors have different index sets (an
    /// index-set mismatch) or the arithmetic reports a failure.
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor_a: TensorDynLen = TensorDynLen::from_dense(indices_a, data_a).unwrap();
    ///
    /// let indices_b = vec![i.clone(), j.clone()];
    /// let data_b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    /// let tensor_b: TensorDynLen = TensorDynLen::from_dense(indices_b, data_b).unwrap();
    ///
    /// let sum = tensor_a.add(&tensor_b).unwrap();
    /// // sum = [[2, 3, 4], [5, 6, 7]]
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
        // Validate that both tensors have the same number of indices
        if self.indices.len() != other.indices.len() {
            return Err(anyhow::anyhow!(
                "Index count mismatch: self has {} indices, other has {}",
                self.indices.len(),
                other.indices.len()
            ));
        }

        // Validate that both tensors have the same set of indices
        let self_set: HashSet<_> = self.indices.iter().collect();
        let other_set: HashSet<_> = other.indices.iter().collect();

        if self_set != other_set {
            return Err(anyhow::anyhow!(
                "Index set mismatch: tensors must have the same indices"
            ));
        }

        // Permute other to match self's index order (no-op if already aligned)
        let other_aligned = other.permute_indices(&self.indices)?;

        // Validate dimensions match after alignment
        let self_expected_dims = Self::expected_dims_from_indices(&self.indices);
        let other_expected_dims = Self::expected_dims_from_indices(&other_aligned.indices);
        if self_expected_dims != other_expected_dims {
            use crate::TagSetLike;
            let fmt = |indices: &[DynIndex]| -> Vec<String> {
                indices
                    .iter()
                    .map(|idx| {
                        let tags: Vec<String> = idx.tags().iter().collect();
                        format!("{:?}(dim={},tags={:?})", idx.id(), idx.dim(), tags)
                    })
                    .collect()
            };
            return Err(anyhow::anyhow!(
                "Dimension mismatch after alignment.\n\
                 self: dims={:?}, indices(order)={:?}\n\
                 other_aligned: dims={:?}, indices(order)={:?}",
                self_expected_dims,
                fmt(&self.indices),
                other_expected_dims,
                fmt(&other_aligned.indices)
            ));
        }

        self.axpby(
            AnyScalar::new_real(1.0),
            &other_aligned,
            AnyScalar::new_real(1.0),
        )
    }

    /// Compute a linear combination: `a * self + b * other`.
    ///
    /// Both tensors must have the same set of indices (matched by ID).
    /// If indices are in a different order, `other` is automatically permuted
    /// to match `self`.
    ///
    /// # Errors
    /// Returns an error when the tensors have different index sets (an index-set
    /// mismatch) or the arithmetic reports a failure.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 4.0]).unwrap();
    ///
    /// // 2*a + 3*b = [2+9, 4+12] = [11, 16]
    /// let result = a.axpby(AnyScalar::new_real(2.0), &b, AnyScalar::new_real(3.0)).unwrap();
    /// let data = result.to_vec::<f64>().unwrap();
    /// assert!((data[0] - 11.0).abs() < 1e-12);
    /// assert!((data[1] - 16.0).abs() < 1e-12);
    /// ```
    pub fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        // Validate that both tensors have the same number of indices.
        if self.indices.len() != other.indices.len() {
            return Err(anyhow::anyhow!(
                "Index count mismatch: self has {} indices, other has {}",
                self.indices.len(),
                other.indices.len()
            ));
        }

        // Validate that both tensors have the same set of indices.
        let self_set: HashSet<_> = self.indices.iter().collect();
        let other_set: HashSet<_> = other.indices.iter().collect();
        if self_set != other_set {
            return Err(anyhow::anyhow!(
                "Index set mismatch: tensors must have the same indices"
            ));
        }

        // Align other tensor axis order to self.
        let other_aligned = other.permute_indices(&self.indices)?;

        // Validate dimensions match after alignment.
        let self_expected_dims = Self::expected_dims_from_indices(&self.indices);
        let other_expected_dims = Self::expected_dims_from_indices(&other_aligned.indices);
        if self_expected_dims != other_expected_dims {
            return Err(anyhow::anyhow!(
                "Dimension mismatch after alignment: self={:?}, other_aligned={:?}",
                self_expected_dims,
                other_expected_dims
            ));
        }

        let axis_classes = if self.storage.axis_classes() == other_aligned.storage.axis_classes() {
            self.storage.axis_classes().to_vec()
        } else {
            Self::dense_axis_classes(self.indices.len())
        };

        let same_compact_layout = self.storage.payload_dims()
            == other_aligned.storage.payload_dims()
            && self.storage.payload_strides_vec() == other_aligned.storage.payload_strides_vec()
            && self.storage.axis_classes() == other_aligned.storage.axis_classes();
        if same_compact_layout
            && matches!(&self.storage, TensorDynLenStorage::Materialized(_))
            && matches!(&other_aligned.storage, TensorDynLenStorage::Materialized(_))
            && !self.tracks_grad()
            && !other_aligned.tracks_grad()
            && !a.tracks_grad()
            && !b.tracks_grad()
        {
            let lhs_storage = self.storage.materialize(self.indices.len())?;
            let rhs_storage = other_aligned
                .storage
                .materialize(other_aligned.indices.len())?;
            let combined = lhs_storage
                .axpby(
                    &a.to_backend_scalar(),
                    rhs_storage.as_ref(),
                    &b.to_backend_scalar(),
                )
                .map_err(|e| anyhow::anyhow!("storage axpby failed: {e}"))?;
            return Self::from_storage(self.indices.clone(), Arc::new(combined));
        }

        let self_native = self.as_native()?;
        let other_native = other_aligned.as_native()?;
        if self_native.dtype() == other_native.dtype()
            && matches!(self_native.dtype(), DType::F32 | DType::C32)
        {
            let lhs = self.scale(a)?;
            let rhs = other_aligned.scale(b)?;
            let combined = lhs
                .try_materialized_inner()?
                .add(rhs.try_materialized_inner()?)
                .map_err(|e| anyhow::anyhow!("tensor addition failed: {e}"))?;
            return Self::from_inner_with_axis_classes(
                self.indices.clone(),
                combined,
                axis_classes,
            );
        }

        let a_native = a.as_tensor()?.as_native()?;
        let b_native = b.as_tensor()?.as_native()?;
        if self_native.dtype() != other_native.dtype()
            || self_native.dtype() != a_native.dtype()
            || other_native.dtype() != b_native.dtype()
        {
            let combined = axpby_native_tensor(
                self_native,
                &a.to_backend_scalar(),
                other_native,
                &b.to_backend_scalar(),
            )?;
            return Self::from_native_with_axis_classes(
                self.indices.clone(),
                combined,
                axis_classes,
            );
        }

        let lhs = self.scale(a)?;
        let rhs = other_aligned.scale(b)?;
        let combined = lhs
            .try_materialized_inner()?
            .add(rhs.try_materialized_inner()?)
            .map_err(|e| anyhow::anyhow!("tensor addition failed: {e}"))?;
        Self::from_inner_with_axis_classes(self.indices.clone(), combined, axis_classes)
    }

    /// Scalar multiplication.
    ///
    /// Multiplies every element by `scalar`.
    ///
    /// # Errors
    /// Returns an error when the scalar coefficient is invalid for the tensor's
    /// scalar type (an invalid scalar dtype) or the backend reports a
    /// failure.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let t = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0, 3.0]).unwrap();
    /// let scaled = t.scale(AnyScalar::new_real(2.0)).unwrap();
    /// assert_eq!(scaled.to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
    /// ```
    pub fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        if matches!(
            &self.storage,
            TensorDynLenStorage::Eager { .. }
                | TensorDynLenStorage::Compact(_)
                | TensorDynLenStorage::Materialized(_)
        ) {
            // Scale via the compact payload only. Materialized structured
            // storage is converted payload-coordinate by payload-coordinate
            // (never the logical domain) and returned as compact storage, so
            // scaling never touches unreferenced strided-gap backing entries.
            let storage = self.storage.scale_eager_payload(&scalar)?;
            return Ok(Self {
                indices: self.indices.clone(),
                storage,
                eager_cache: Self::empty_eager_cache(),
            });
        }

        let self_native = self.as_native()?;
        let scalar_native = scalar.as_tensor()?.as_native()?;
        if matches!(self_native.dtype(), DType::F32 | DType::C32)
            && self_native.dtype() != scalar_native.dtype()
        {
            let target_dtype =
                Self::scale_target_dtype(self_native.dtype(), scalar_native.dtype())?;
            let self_inner = self.try_materialized_inner()?;
            let self_inner = if self_inner.data().dtype() == target_dtype {
                self_inner.clone()
            } else {
                self_inner.convert(target_dtype)?
            };
            let scalar_inner = scalar.as_tensor()?.try_materialized_inner()?;
            let scalar_inner = if scalar_inner.data().dtype() == target_dtype {
                scalar_inner.clone()
            } else {
                scalar_inner.convert(target_dtype)?
            };
            let scaled = if self.indices.is_empty() {
                self_inner
                    .mul(&scalar_inner)
                    .map_err(|e| anyhow::anyhow!("scalar multiplication failed: {e}"))?
            } else {
                let subscripts = Self::scale_subscripts(self.indices.len())?;
                eager_einsum_ad(&[&self_inner, &scalar_inner], &subscripts)
                    .map_err(|e| anyhow::anyhow!("tensor scaling failed: {e}"))?
            };
            return Self::from_inner_with_axis_classes(
                self.indices.clone(),
                scaled,
                self.storage.axis_classes().to_vec(),
            );
        }
        if self_native.dtype() != scalar_native.dtype() {
            let scaled = scale_native_tensor(self_native, &scalar.to_backend_scalar())?;
            return Self::from_native_with_axis_classes(
                self.indices.clone(),
                scaled,
                self.storage.axis_classes().to_vec(),
            );
        }

        let scaled = if self.indices.is_empty() {
            self.try_materialized_inner()?
                .mul(scalar.as_tensor()?.try_materialized_inner()?)
                .map_err(|e| anyhow::anyhow!("scalar multiplication failed: {e}"))?
        } else {
            let subscripts = Self::scale_subscripts(self.indices.len())?;
            eager_einsum_ad(
                &[
                    self.try_materialized_inner()?,
                    scalar.as_tensor()?.try_materialized_inner()?,
                ],
                &subscripts,
            )
            .map_err(|e| anyhow::anyhow!("tensor scaling failed: {e}"))?
        };
        Self::from_inner_with_axis_classes(
            self.indices.clone(),
            scaled,
            self.storage.axis_classes().to_vec(),
        )
    }

    /// Inner product (dot product) of two tensors.
    ///
    /// Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.
    ///
    /// # Errors
    /// Returns an error when the tensors have different index sets (an index-set
    /// mismatch) or the contraction reports a failure.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    /// let b = TensorDynLen::from_dense(vec![i.clone()], vec![4.0, 5.0, 6.0]).unwrap();
    ///
    /// // <a, b> = 1*4 + 2*5 + 3*6 = 32
    /// let ip = a.inner_product(&b).unwrap();
    /// assert!((ip.real() - 32.0).abs() < 1e-12);
    /// ```
    pub fn inner_product(&self, other: &Self) -> Result<AnyScalar> {
        if self.indices.len() == other.indices.len() {
            let self_set: HashSet<_> = self.indices.iter().collect();
            let other_set: HashSet<_> = other.indices.iter().collect();
            if self_set == other_set {
                let other_aligned = other.permute_indices(&self.indices)?;
                let result = super::contract::contract_pair_with_operand_options(
                    self,
                    &other_aligned,
                    PairwiseContractionOptions::new().with_lhs_conj(true),
                )?;
                return result.sum();
            }
        }

        // Contract self.conj() with other over all indices
        let result = super::contract::contract_pair_with_operand_options(
            self,
            other,
            PairwiseContractionOptions::new().with_lhs_conj(true),
        )?;
        // Result should be a scalar (no indices)
        result.sum()
    }
}

// ============================================================================
// Index Replacement Methods
// ============================================================================

impl TensorDynLen {
    /// Replace an index in the tensor with a new index.
    ///
    /// This replaces the index matching `old_index` by ID with `new_index`.
    /// The storage data is not modified, only the index metadata is changed.
    ///
    /// # Arguments
    /// * `old_index` - The index to replace (matched by ID)
    /// * `new_index` - The new index to use
    ///
    /// # Returns
    /// A new tensor with the index replaced. If no index matches `old_index`,
    /// returns a clone of the original tensor.
    ///
    /// # Errors
    /// Returns an error when `old_index` is not present (a missing-index failure)
    /// or the new index has an incompatible dimension (a dimension mismatch).
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);  // Same dimension, different ID
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Replace index i with new_i
    /// let replaced = tensor.replaceind(&i, &new_i).unwrap();
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, j.id);
    /// ```
    pub fn replaceind(&self, old_index: &DynIndex, new_index: &DynIndex) -> Result<Self> {
        // Validate dimension match
        if old_index.dim() != new_index.dim() {
            return Err(anyhow::anyhow!(
                "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                old_index.dim(),
                new_index.dim()
            ));
        }

        let new_indices: Vec<_> = self
            .indices
            .iter()
            .map(|idx| {
                if *idx == *old_index {
                    new_index.clone()
                } else {
                    idx.clone()
                }
            })
            .collect();

        Ok(Self {
            indices: new_indices,
            storage: self.storage.clone(),
            eager_cache: Arc::clone(&self.eager_cache),
        })
    }

    /// Replace multiple indices in the tensor.
    ///
    /// This replaces each index in `old_indices` (matched by ID) with the corresponding
    /// index in `new_indices`. The storage data is not modified.
    ///
    /// # Arguments
    /// * `old_indices` - The indices to replace (matched by ID)
    /// * `new_indices` - The new indices to use
    ///
    /// # Returns
    /// A new tensor with the indices replaced. Indices not found in `old_indices`
    /// are kept unchanged.
    ///
    /// # Errors
    /// Returns an error when an `old_indices` entry is not present (a
    /// missing-index failure) or `old_indices`/`new_indices` differ in length
    /// (a length mismatch).
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);
    /// let new_j = Index::new_dyn(3);
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Replace both indices
    /// let replaced = tensor
    ///     .replaceinds(&[i.clone(), j.clone()], &[new_i.clone(), new_j.clone()])
    ///     .unwrap();
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, new_j.id);
    /// ```
    pub fn replaceinds(&self, old_indices: &[DynIndex], new_indices: &[DynIndex]) -> Result<Self> {
        anyhow::ensure!(
            old_indices.len() == new_indices.len(),
            "old_indices and new_indices must have the same length"
        );

        // Validate dimension matches for all replacements
        for (old, new) in old_indices.iter().zip(new_indices.iter()) {
            if old.dim() != new.dim() {
                return Err(anyhow::anyhow!(
                    "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                    old.dim(),
                    new.dim()
                ));
            }
        }

        // Build a map from old indices to new indices
        let replacement_map: std::collections::HashMap<_, _> =
            old_indices.iter().zip(new_indices.iter()).collect();

        let new_indices_vec: Vec<_> = self
            .indices
            .iter()
            .map(|idx| {
                if let Some(new_idx) = replacement_map.get(idx) {
                    (*new_idx).clone()
                } else {
                    idx.clone()
                }
            })
            .collect();

        Ok(Self {
            indices: new_indices_vec,
            storage: self.storage.clone(),
            eager_cache: Arc::clone(&self.eager_cache),
        })
    }
}

// ============================================================================
// Complex Conjugation
// ============================================================================

impl TensorDynLen {
    /// Complex conjugate of all tensor elements.
    ///
    /// For real (`f32`/`f64`) tensors, returns a copy (conjugate of real is
    /// identity). For complex (`Complex32`/`Complex64`) tensors, conjugates
    /// each element.
    ///
    /// The indices and dimensions remain unchanged. If an eager backend cannot
    /// perform the conjugation, the failure is retained and reported by the
    /// next fallible materialization or AD-sensitive operation.
    ///
    /// This is inspired by the `conj` operation in ITensorMPS.jl.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use num_complex::Complex64;
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i], data).unwrap();
    ///
    /// let conj_tensor = tensor.conj();
    /// assert_eq!(
    ///     conj_tensor.to_vec::<Complex64>().unwrap(),
    ///     vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)]
    /// );
    /// ```
    pub fn conj(&self) -> Self {
        self.conj_with(&conjugate_eager)
    }

    fn conj_with<F>(&self, conjugate: &F) -> Self
    where
        F: Fn(
            &EagerTensor,
        ) -> std::result::Result<
            EagerTensor,
            Arc<dyn std::error::Error + Send + Sync + 'static>,
        >,
    {
        // Conjugate tensor storage and map indices via IndexLike::conj(). For
        // default undirected indices, conj() is a no-op; this remains future-
        // proof for QSpace-compatible directed indices.
        let new_indices: Vec<DynIndex> = self.indices.iter().map(|idx| idx.conj()).collect();
        let mut storage = match self.storage.conjugate_with(conjugate) {
            Ok(storage) => storage,
            Err(error) => self.storage.clone().with_deferred_error(error),
        };
        let mut eager_cache = if storage.deferred_error().is_some() {
            Arc::clone(&self.eager_cache)
        } else {
            Self::empty_eager_cache()
        };

        if storage.deferred_error().is_none() {
            if let Some(inner) = self.eager_cache.get() {
                match conjugate(inner.as_ref()) {
                    Ok(conjugated) => eager_cache = Self::eager_cache_with(conjugated),
                    Err(source) => {
                        storage =
                            storage.with_deferred_error(TensorStorageError::Conjugation { source });
                        // Keep the original cache alive so a deferred failure
                        // retains its graph until a fallible consumer reports it.
                        eager_cache = Arc::clone(&self.eager_cache);
                    }
                }
            }
        }

        Self {
            indices: new_indices,
            storage,
            eager_cache,
        }
    }
}

#[derive(Debug, Default)]
struct Lassq {
    scale: f64,
    sumsq: f64,
    infinite: bool,
}

impl Lassq {
    fn add_component(&mut self, value: f64) {
        let value = value.abs();
        if value == 0.0 {
            return;
        }
        if value.is_infinite() {
            self.infinite = true;
            return;
        }
        if self.scale < value {
            if self.scale == 0.0 {
                self.sumsq = 1.0;
            } else {
                let ratio = self.scale / value;
                self.sumsq = 1.0 + self.sumsq * ratio * ratio;
            }
            self.scale = value;
        } else {
            let ratio = value / self.scale;
            self.sumsq += ratio * ratio;
        }
    }

    fn add_complex(&mut self, value: Complex64) {
        self.add_component(value.re);
        self.add_component(value.im);
    }

    fn add_scaled(&mut self, scale: f64, coefficient: f64) {
        if scale == 0.0 || coefficient == 0.0 {
            return;
        }
        if self.scale < scale {
            if self.scale == 0.0 {
                self.sumsq = coefficient * coefficient;
            } else {
                let ratio = self.scale / scale;
                self.sumsq = coefficient * coefficient + self.sumsq * ratio * ratio;
            }
            self.scale = scale;
        } else {
            let ratio = scale / self.scale * coefficient;
            self.sumsq += ratio * ratio;
        }
    }

    fn add_component_difference(&mut self, lhs: f64, rhs: f64) {
        if lhs == rhs {
            return;
        }
        let scale = lhs.abs().max(rhs.abs());
        if scale != 0.0 {
            self.add_scaled(scale, (lhs / scale - rhs / scale).abs());
        }
    }

    fn add_complex_difference(&mut self, lhs: Complex64, rhs: Complex64) {
        self.add_component_difference(lhs.re, rhs.re);
        self.add_component_difference(lhs.im, rhs.im);
    }

    fn is_zero(&self) -> bool {
        !self.infinite && self.scale == 0.0
    }

    fn norm(&self) -> f64 {
        if self.infinite {
            f64::INFINITY
        } else if self.scale == 0.0 {
            0.0
        } else {
            self.scale * self.sumsq.sqrt()
        }
    }

    fn norm_squared(&self) -> f64 {
        let norm = self.norm();
        norm * norm
    }

    fn log_norm(&self) -> f64 {
        if self.infinite {
            f64::INFINITY
        } else if self.scale == 0.0 {
            f64::NEG_INFINITY
        } else {
            self.scale.ln() + 0.5 * self.sumsq.ln()
        }
    }
}

// ============================================================================
// Norm Computation
// ============================================================================

impl TensorDynLen {
    /// Compute the squared Frobenius norm of the tensor: ||T||² = Σ|T_ijk...|²
    ///
    /// For real tensors: sum of squares of all elements.
    /// For complex tensors: sum of `|z|²` over the compact payload.
    /// The reduction promotes source values to `f64` and uses a stable LASSQ
    /// accumulator, so it does not form a source-dtype `self * conj(self)`.
    ///
    /// # Errors
    /// Returns [`TensorDynLenError`] when storage/materialization or scalar
    /// extraction fails, or when the input produces NaN. The result is
    /// accumulated from squared magnitudes, so it is never negative;
    /// positive infinity is preserved.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 1² + 2² + ... + 6² = 91
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i, j], data).unwrap();
    ///
    /// assert!((tensor.norm_squared().unwrap() - 91.0).abs() < 1e-10);
    /// ```
    pub fn norm_squared(&self) -> std::result::Result<f64, TensorDynLenError> {
        let dtype = self
            .scalar_dtype()
            .map_err(TensorDynLenError::scalar_extraction)?;
        if !matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64) {
            return Err(TensorDynLenError::ScalarTypeMismatch {
                expected: "f32, f64, c32, or c64",
                actual: Self::dtype_name(dtype).to_string(),
            });
        }
        let (has_nan, _) = self
            .compact_nonfinite_flags()
            .map_err(TensorDynLenError::materialization)?;
        if has_nan {
            return Err(TensorDynLenError::NaNInput {
                operation: "norm_squared",
            });
        }

        let mut norm = Lassq::default();
        self.storage
            .for_each_payload_value(|value| norm.add_complex(value))
            .map_err(TensorDynLenError::materialization)?;
        let value = norm.norm_squared();
        if value.is_nan() {
            return Err(TensorDynLenError::NaNInput {
                operation: "norm_squared",
            });
        }
        Ok(value)
    }

    /// Compute the Frobenius norm of the tensor: ||T|| = sqrt(Σ|T_ijk...|²)
    ///
    /// # Errors
    /// Returns [`TensorDynLenError`] when norm evaluation fails or when the
    /// input contains NaN. Positive infinity is preserved.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![3.0, 4.0];  // sqrt(9 + 16) = 5
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i], data).unwrap();
    ///
    /// assert!((tensor.norm().unwrap() - 5.0).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> std::result::Result<f64, TensorDynLenError> {
        Ok(self.norm_squared()?.sqrt())
    }

    /// Maximum absolute value of all elements (L-infinity norm).
    ///
    /// # Errors
    /// Returns [`TensorDynLenError`] when authoritative storage or eager
    /// materialization cannot be read, or when the input contains NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(4);
    /// let t = TensorDynLen::from_dense(vec![i], vec![-5.0, 1.0, 3.0, -2.0]).unwrap();
    /// assert!((t.maxabs().unwrap() - 5.0).abs() < 1e-12);
    /// ```
    pub fn maxabs(&self) -> std::result::Result<f64, TensorDynLenError> {
        if let Some(error) = self.storage.deferred_error() {
            return Err(TensorDynLenError::Storage {
                source: error.clone(),
            });
        }
        let dtype = self
            .storage
            .dtype()
            .ok_or_else(|| TensorDynLenError::ScalarTypeMismatch {
                expected: "f32, f64, c32, or c64",
                actual: "unknown".to_string(),
            })?;
        if !matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64) {
            return Err(TensorDynLenError::ScalarTypeMismatch {
                expected: "f32, f64, c32, or c64",
                actual: Self::dtype_name(dtype).to_string(),
            });
        }
        let (has_nan, _) = self
            .compact_nonfinite_flags()
            .map_err(TensorDynLenError::materialization)?;
        if has_nan {
            return Err(TensorDynLenError::NaNInput {
                operation: "maxabs",
            });
        }
        let mut value = 0.0_f64;
        self.storage
            .for_each_payload_value(|scalar| {
                let magnitude = scalar.re.hypot(scalar.im);
                value = value.max(magnitude);
            })
            .map_err(TensorDynLenError::materialization)?;
        Ok(value)
    }

    fn native_complex_payload_value_at(
        native: &NativeTensor,
        payload_coords: &[usize],
    ) -> Result<Complex64> {
        anyhow::ensure!(
            payload_coords.len() == native.shape().len(),
            "payload coordinate rank {} does not match payload rank {}",
            payload_coords.len(),
            native.shape().len()
        );
        let mut offset = 0usize;
        let mut stride = 1usize;
        for (&coordinate, &dim) in payload_coords.iter().zip(native.shape().iter()) {
            anyhow::ensure!(
                coordinate < dim,
                "payload coordinate {coordinate} is out of bounds for dim {dim}"
            );
            offset = offset
                .checked_add(
                    coordinate
                        .checked_mul(stride)
                        .ok_or_else(|| anyhow::anyhow!("payload offset overflow"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("payload offset overflow"))?;
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("payload stride overflow"))?;
        }
        match native.dtype() {
            DType::F32 => native
                .as_slice::<f32>()
                .and_then(|values| values.get(offset).copied())
                .map(|value| Complex64::new(f64::from(value), 0.0))
                .ok_or_else(|| anyhow::anyhow!("failed to read f32 payload value")),
            DType::F64 => native
                .as_slice::<f64>()
                .and_then(|values| values.get(offset).copied())
                .map(|value| Complex64::new(value, 0.0))
                .ok_or_else(|| anyhow::anyhow!("failed to read f64 payload value")),
            DType::C32 => native
                .as_slice::<Complex32>()
                .and_then(|values| values.get(offset).copied())
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .ok_or_else(|| anyhow::anyhow!("failed to read c32 payload value")),
            DType::C64 => native
                .as_slice::<Complex64>()
                .and_then(|values| values.get(offset).copied())
                .ok_or_else(|| anyhow::anyhow!("failed to read c64 payload value")),
            dtype => Err(anyhow::anyhow!("unsupported payload dtype {dtype:?}")),
        }
    }

    fn native_sum_scalar(native: &NativeTensor) -> Result<AnyScalar> {
        match native.dtype() {
            DType::F32 => native
                .as_slice::<f32>()
                .map(|values| AnyScalar::from_value(values.iter().copied().sum::<f32>()))
                .ok_or_else(|| anyhow::anyhow!("failed to read f32 payload")),
            DType::F64 => native
                .as_slice::<f64>()
                .map(|values| AnyScalar::from_value(values.iter().copied().sum::<f64>()))
                .ok_or_else(|| anyhow::anyhow!("failed to read f64 payload")),
            DType::C32 => native
                .as_slice::<Complex32>()
                .map(|values| AnyScalar::from_value(values.iter().copied().sum::<Complex32>()))
                .ok_or_else(|| anyhow::anyhow!("failed to read c32 payload")),
            DType::C64 => native
                .as_slice::<Complex64>()
                .map(|values| AnyScalar::from_value(values.iter().copied().sum::<Complex64>()))
                .ok_or_else(|| anyhow::anyhow!("failed to read c64 payload")),
            dtype => Err(anyhow::anyhow!("unsupported dtype {dtype:?}")),
        }
    }

    fn native_nonfinite_flags(native: &NativeTensor) -> Result<(bool, bool)> {
        let mut has_nan = false;
        let mut has_infinity = false;
        match native.dtype() {
            DType::F32 => {
                let values = native
                    .as_slice::<f32>()
                    .ok_or_else(|| anyhow::anyhow!("failed to read f32 payload"))?;
                for &value in values {
                    has_nan |= value.is_nan();
                    has_infinity |= value.is_infinite();
                }
            }
            DType::F64 => {
                let values = native
                    .as_slice::<f64>()
                    .ok_or_else(|| anyhow::anyhow!("failed to read f64 payload"))?;
                for &value in values {
                    has_nan |= value.is_nan();
                    has_infinity |= value.is_infinite();
                }
            }
            DType::C32 => {
                let values = native
                    .as_slice::<Complex32>()
                    .ok_or_else(|| anyhow::anyhow!("failed to read c32 payload"))?;
                for value in values {
                    has_nan |= value.re.is_nan() || value.im.is_nan();
                    has_infinity |= value.re.is_infinite() || value.im.is_infinite();
                }
            }
            DType::C64 => {
                let values = native
                    .as_slice::<Complex64>()
                    .ok_or_else(|| anyhow::anyhow!("failed to read c64 payload"))?;
                for value in values {
                    has_nan |= value.re.is_nan() || value.im.is_nan();
                    has_infinity |= value.re.is_infinite() || value.im.is_infinite();
                }
            }
            dtype => return Err(anyhow::anyhow!("unsupported dtype {dtype:?}")),
        }
        Ok((has_nan, has_infinity))
    }

    fn compact_nonfinite_flags(&self) -> Result<(bool, bool)> {
        self.storage.nonfinite_flags()
    }

    /// Element-wise subtraction with index alignment.
    ///
    /// This computes `self - other` using the same vector-space semantics as
    /// [`TensorVectorSpace`](crate::TensorVectorSpace).
    ///
    /// # Errors
    /// Returns an error when the tensors have different index sets (an index-set
    /// mismatch) or the arithmetic reports a failure.
    ///
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.axpby(AnyScalar::new_real(1.0), other, AnyScalar::new_real(-1.0))
    }

    /// Negate all elements.
    ///
    /// # Errors
    /// Returns an error when scalar multiplication fails for the tensor storage
    /// (a dtype mismatch) or the backend reports a failure.
    ///
    pub fn neg(&self) -> Result<Self> {
        self.scale(AnyScalar::new_real(-1.0))
    }

    /// Approximate equality check using Julia `isapprox`-style semantics.
    ///
    /// Values are aligned by index identity and streamed from each tensor's
    /// compact support. Exact zero-tolerance comparisons use exact scalar
    /// equality; nonzero tolerances use scaled sum-of-squares accumulation,
    /// avoiding logical-dense traversal and avoidable underflow/overflow.
    ///
    /// # Errors
    /// Returns [`TensorDynLenError`] when tolerances are invalid, the index
    /// spaces cannot be aligned, storage cannot be read, or either input
    /// contains NaN.
    pub fn isapprox(
        &self,
        other: &Self,
        atol: f64,
        rtol: f64,
    ) -> std::result::Result<bool, TensorDynLenError> {
        for (name, value) in [("atol", atol), ("rtol", rtol)] {
            if !value.is_finite() || value < 0.0 {
                return Err(TensorDynLenError::InvalidTolerance { name, value });
            }
        }
        if self.indices.len() != other.indices.len() {
            return Err(TensorDynLenError::ShapeMismatch {
                operation: "isapprox",
                expected: format!("indices {:?}", self.indices),
                actual: format!("indices {:?}", other.indices),
            });
        }

        let other_axis_by_index = other
            .indices
            .iter()
            .cloned()
            .enumerate()
            .map(|(axis, index)| (index, axis))
            .collect::<HashMap<_, _>>();
        let other_positions = self
            .indices
            .iter()
            .map(|index| {
                other_axis_by_index.get(index).copied().ok_or_else(|| {
                    TensorDynLenError::ShapeMismatch {
                        operation: "isapprox",
                        expected: format!("indices {:?}", self.indices),
                        actual: format!("indices {:?}", other.indices),
                    }
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let self_dims = self.dims();
        let other_dims = other.dims();
        for (axis, &other_axis) in other_positions.iter().enumerate() {
            if self_dims[axis] != other_dims[other_axis] {
                return Err(TensorDynLenError::ShapeMismatch {
                    operation: "isapprox",
                    expected: format!("dims {:?}", self_dims),
                    actual: format!("dims {:?}", other_dims),
                });
            }
        }
        for tensor in [self, other] {
            if tensor
                .compact_nonfinite_flags()
                .map_err(TensorDynLenError::materialization)?
                .0
            {
                return Err(TensorDynLenError::NaNInput {
                    operation: "isapprox",
                });
            }
        }

        let exact = atol == 0.0 && rtol == 0.0;
        let lhs_payload_dims = self.storage.payload_dims().to_vec();
        let rhs_payload_dims = other.storage.payload_dims().to_vec();
        let lhs_payload_len =
            checked_product(&lhs_payload_dims).map_err(TensorDynLenError::materialization)?;
        let rhs_payload_len =
            checked_product(&rhs_payload_dims).map_err(TensorDynLenError::materialization)?;
        let lhs_axis_classes = self.storage.axis_classes();
        let rhs_axis_classes = other.storage.axis_classes();
        let self_to_other = other_positions.clone();
        let mut other_to_self = vec![0usize; self_to_other.len()];
        for (self_axis, &other_axis) in self_to_other.iter().enumerate() {
            other_to_self[other_axis] = self_axis;
        }

        let mut diff = Lassq::default();
        let mut lhs_norm = Lassq::default();
        let mut rhs_norm = Lassq::default();
        let mut compare = |lhs: Complex64, rhs: Complex64| -> bool {
            if exact {
                return lhs == rhs;
            }
            let lhs_infinite = lhs.re.is_infinite() || lhs.im.is_infinite();
            let rhs_infinite = rhs.re.is_infinite() || rhs.im.is_infinite();
            if lhs_infinite || rhs_infinite {
                return lhs == rhs;
            }
            lhs_norm.add_complex(lhs);
            rhs_norm.add_complex(rhs);
            diff.add_complex_difference(lhs, rhs);
            true
        };

        // Each compact payload coordinate identifies exactly one logical
        // support point. Map it through the aligned logical axes instead of
        // traversing structural zeros in the logical tensor.
        let mut lhs_coords = vec![0usize; lhs_payload_dims.len()];
        let mut rhs_from_lhs = vec![0usize; rhs_payload_dims.len()];
        let mut rhs_seen = vec![false; rhs_payload_dims.len()];
        for _ in 0..lhs_payload_len {
            let lhs = self
                .storage
                .payload_value_at(&lhs_coords)
                .map_err(TensorDynLenError::materialization)?;
            if map_payload_support_coordinate(
                lhs_axis_classes,
                rhs_axis_classes,
                &self_to_other,
                &lhs_coords,
                &mut rhs_from_lhs,
                &mut rhs_seen,
            ) {
                let rhs = other
                    .storage
                    .payload_value_at(&rhs_from_lhs)
                    .map_err(TensorDynLenError::materialization)?;
                if !compare(lhs, rhs) {
                    return Ok(false);
                }
            } else if !compare(lhs, Complex64::new(0.0, 0.0)) {
                return Ok(false);
            }

            increment_col_major_coordinate(&mut lhs_coords, &lhs_payload_dims);
        }

        // The reverse pass accounts for support points that exist only in the
        // right tensor. Overlap points were compared in the first pass and are
        // therefore skipped here without a payload-sized visited set.
        let mut rhs_coords = vec![0usize; rhs_payload_dims.len()];
        let mut lhs_from_rhs = vec![0usize; lhs_payload_dims.len()];
        let mut lhs_seen = vec![false; lhs_payload_dims.len()];
        for _ in 0..rhs_payload_len {
            let rhs = other
                .storage
                .payload_value_at(&rhs_coords)
                .map_err(TensorDynLenError::materialization)?;
            if !map_payload_support_coordinate(
                rhs_axis_classes,
                lhs_axis_classes,
                &other_to_self,
                &rhs_coords,
                &mut lhs_from_rhs,
                &mut lhs_seen,
            ) && !compare(Complex64::new(0.0, 0.0), rhs)
            {
                return Ok(false);
            }
            increment_col_major_coordinate(&mut rhs_coords, &rhs_payload_dims);
        }

        if exact {
            return Ok(true);
        }
        let absolute_ok = if diff.is_zero() {
            true
        } else if atol == 0.0 || diff.infinite {
            false
        } else {
            diff.log_norm() <= atol.ln()
        };
        let relative_ok = if rtol == 0.0 || diff.is_zero() {
            diff.is_zero()
        } else if diff.infinite {
            lhs_norm.infinite || rhs_norm.infinite
        } else if lhs_norm.infinite || rhs_norm.infinite {
            true
        } else {
            let reference_log = lhs_norm.log_norm().max(rhs_norm.log_norm());
            diff.log_norm() <= rtol.ln() + reference_log
        };
        Ok(absolute_ok || relative_ok)
    }

    /// Create a diagonal Kronecker-delta tensor for one input/output index pair.
    ///
    /// # Errors
    /// Returns an error when the two indices have different dimensions (an
    /// index dimension mismatch).
    ///
    pub fn diagonal(input_index: &DynIndex, output_index: &DynIndex) -> Result<Self> {
        <Self as TensorConstructionLike>::diagonal(input_index, output_index)
            .map_err(anyhow::Error::new)
    }

    /// Create a product of Kronecker-delta tensors for paired index lists.
    ///
    /// # Errors
    /// Returns an error if the index lists have different lengths or paired
    /// dimensions do not match.
    pub fn delta(input_indices: &[DynIndex], output_indices: &[DynIndex]) -> Result<Self> {
        <Self as TensorConstructionLike>::delta(input_indices, output_indices)
            .map_err(anyhow::Error::new)
    }

    /// Create a scalar tensor equal to one.
    ///
    /// # Errors
    /// Returns an error when dense scalar construction fails for the element type
    /// (an invalid scalar dtype or a construction failure).
    ///
    pub fn scalar_one() -> Result<Self> {
        <Self as TensorConstructionLike>::scalar_one().map_err(anyhow::Error::new)
    }

    /// Create a tensor filled with ones over the given indices.
    ///
    /// # Errors
    /// Returns an error when the tensor size overflows (an overflow failure) or
    /// dense construction fails.
    ///
    pub fn ones(indices: &[DynIndex]) -> Result<Self> {
        <Self as TensorConstructionLike>::ones(indices).map_err(anyhow::Error::new)
    }

    /// Create a one-hot tensor with value one at the specified index positions.
    ///
    /// # Errors
    /// Returns an error when any coordinate is outside its index dimension (an
    /// out-of-bounds failure).
    ///
    pub fn onehot(index_vals: &[(DynIndex, usize)]) -> Result<Self> {
        <Self as TensorConstructionLike>::onehot(index_vals).map_err(anyhow::Error::new)
    }

    /// Keep one coordinate along an index while retaining that index axis.
    ///
    /// This is the differentiable masking counterpart to [`Self::select_indices`].
    /// It selects the requested slice, forms a one-hot tensor over the removed
    /// axis in the source dtype, and takes an explicit tensor product to restore
    /// the original index order. The implementation stays in the tensor backend,
    /// so structured storage and reverse-mode metadata are preserved whenever
    /// the backend can represent the operation.
    ///
    /// # Arguments
    ///
    /// * `index` - Existing tensor index to mask.
    /// * `position` - Zero-based coordinate to keep; all other coordinates become
    ///   zero.
    ///
    /// # Errors
    /// Returns an error when the coordinate is outside the index dimension (an
    /// out-of-bounds failure) or the mask construction fails.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap();
    /// let masked = tensor.mask_index(&i, 1).unwrap();
    ///
    /// assert_eq!(masked.indices(), &[i]);
    /// assert_eq!(masked.to_vec::<f64>().unwrap(), vec![0.0, 4.0]);
    /// assert!(TensorDynLen::from_dense(
    ///     vec![DynIndex::new_dyn(2)],
    ///     vec![1.0_f64, 2.0],
    /// )
    /// .unwrap()
    /// .mask_index(&DynIndex::new_dyn(2), 0)
    /// .is_err());
    /// ```
    pub fn mask_index(&self, index: &DynIndex, position: usize) -> Result<Self> {
        anyhow::ensure!(
            self.indices.iter().any(|candidate| candidate == index),
            "mask_index: index is not present in tensor"
        );
        anyhow::ensure!(
            position < index.dim(),
            "mask_index: position {position} is out of range for dimension {}",
            index.dim()
        );

        // Retaining the shared index turns contraction into a backend-level
        // elementwise product instead of materializing a host mask. Construct
        // the constant mask in the input dtype so f32/c32 values and AD graphs
        // are not promoted or detached.
        let mask = match self.scalar_dtype()? {
            DType::F32 => Self::from_dense(
                vec![index.clone()],
                (0..index.dim())
                    .map(|value| if value == position { 1.0_f32 } else { 0.0 })
                    .collect(),
            ),
            DType::F64 => Self::from_dense(
                vec![index.clone()],
                (0..index.dim())
                    .map(|value| if value == position { 1.0_f64 } else { 0.0 })
                    .collect(),
            ),
            DType::C32 => Self::from_dense(
                vec![index.clone()],
                (0..index.dim())
                    .map(|value| {
                        if value == position {
                            num_complex::Complex32::new(1.0, 0.0)
                        } else {
                            num_complex::Complex32::new(0.0, 0.0)
                        }
                    })
                    .collect(),
            ),
            DType::C64 => Self::from_dense(
                vec![index.clone()],
                (0..index.dim())
                    .map(|value| {
                        if value == position {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .collect(),
            ),
            dtype => {
                return Err(anyhow::anyhow!(
                    "mask_index does not support dtype {dtype:?}"
                ))
            }
        }?;
        super::contract::contract_pair_with_options(
            self,
            &mask,
            super::contract::ContractionOptions::new()
                .with_retain_indices(std::slice::from_ref(index)),
        )
    }

    /// Compute the relative distance between two tensors.
    ///
    /// Returns `||A - B|| / ||A||` (Frobenius norm).
    /// If `||A|| = 0`, returns `||B||` instead to avoid division by zero.
    ///
    /// This is the ITensor-style distance function useful for comparing tensors.
    ///
    /// # Arguments
    /// * `other` - The other tensor to compare with
    ///
    /// # Errors
    /// Returns [`TensorDynLenError`] when either norm contains NaN, or when
    /// scaling and subtracting the tensors fails.
    ///
    /// # Returns
    /// The relative distance as a f64 value.
    ///
    /// # Note
    /// The indices of both tensors must be permutable to each other.
    /// The result tensor (A - B) uses the index ordering from self.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let data_a = vec![1.0, 0.0];
    /// let data_b = vec![1.0, 0.0];  // Same tensor
    /// let tensor_a: TensorDynLen = TensorDynLen::from_dense(vec![i.clone()], data_a).unwrap();
    /// let tensor_b: TensorDynLen = TensorDynLen::from_dense(vec![i.clone()], data_b).unwrap();
    ///
    /// assert!(tensor_a.distance(&tensor_b).unwrap() < 1e-10);  // Zero distance
    /// ```
    pub fn distance(&self, other: &Self) -> std::result::Result<f64, TensorDynLenError> {
        let norm_self = self.norm()?;

        // Compute A - B = A + (-1) * B
        let neg_other = other
            .scale(AnyScalar::new_real(-1.0))
            .map_err(|error| TensorDynLenError::operation("distance", error))?;
        let diff = self
            .add(&neg_other)
            .map_err(|error| TensorDynLenError::operation("distance", error))?;
        let norm_diff = diff.norm()?;

        if norm_self > 0.0 {
            Ok(norm_diff / norm_self)
        } else {
            Ok(norm_diff)
        }
    }
}

impl std::fmt::Debug for TensorDynLen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorDynLen")
            .field("indices", &self.indices)
            .field("dims", &self.dims())
            .field("is_diag", &self.is_diag())
            .finish()
    }
}

/// Create a diagonal tensor with dynamic rank from diagonal data.
/// # Arguments
/// * `indices` - The indices for the tensor (all must have the same dimension)
/// * `diag_data` - The diagonal elements (length must equal the dimension of indices)
/// The returned tensor preserves compact diagonal payload metadata; use
/// [`TensorDynLen::is_diag`] or [`TensorDynLen::storage`] to inspect that
/// representation.
/// # Errors
/// Returns an error when the index dimensions are unequal (a dimension
/// mismatch) or the diagonal construction fails.
/// # Panics
/// Panics if indices have different dimensions, or if diag_data length doesn't match.
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, diag_tensor_dyn_len};
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let t = diag_tensor_dyn_len(vec![i, j], vec![1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(t.dims(), vec![3, 3]);
/// assert!(t.is_diag());
/// ```
pub fn diag_tensor_dyn_len(indices: Vec<DynIndex>, diag_data: Vec<f64>) -> Result<TensorDynLen> {
    TensorDynLen::from_diag(indices, diag_data)
}

#[allow(clippy::type_complexity)]
pub(crate) type UnfoldSplitInnerResult = (
    EagerTensor,
    usize,
    usize,
    usize,
    Vec<DynIndex>,
    Vec<DynIndex>,
);

/// Unfold a tensor into a matrix by splitting indices into left and right groups.
/// This function validates the split, permutes the tensor so that left indices
/// come first, and returns a rank-2 native tenferro tensor along with metadata.
/// # Arguments
/// * `t` - Input tensor
/// * `left_inds` - Indices to place on the left (row) side of the matrix
/// # Returns
/// A tuple `(matrix_tensor, left_len, m, n, left_indices, right_indices)` where:
/// - `matrix_tensor` is a rank-2 `tenferro::Tensor` with shape `[m, n]`
/// - `left_len` is the number of left indices
/// - `m` is the product of left index dimensions
/// - `n` is the product of right index dimensions
/// - `left_indices` is the vector of left indices (cloned)
/// - `right_indices` is the vector of right indices (cloned)
/// # Errors
///
/// Returns an error when the tensor rank is less than 2 (a rank mismatch),
/// when `left_inds` is empty or contains all indices (an invalid split), when
/// `left_inds` contains indices not present in the tensor (a missing-index
/// failure) or duplicates, or when the native reshape fails (a backend
/// failure).
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, unfold_split};
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// // 2x3 dense tensor with data [1..6]
/// let t = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ).unwrap();
/// let (matrix, left_len, m, n, left_indices, right_indices) =
///     unfold_split(&t, &[i]).unwrap();
/// assert_eq!(left_len, 1);
/// assert_eq!(m, 2);
/// assert_eq!(n, 3);
/// assert_eq!(left_indices.len(), 1);
/// assert_eq!(right_indices.len(), 1);
/// ```
#[allow(clippy::type_complexity)]
pub fn unfold_split(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(
    NativeTensor,
    usize,
    usize,
    usize,
    Vec<DynIndex>,
    Vec<DynIndex>,
)> {
    let (matrix_inner, left_len, m, n, left_indices, right_indices) =
        unfold_split_inner(t, left_inds)?;

    Ok((
        matrix_inner.data().clone(),
        left_len,
        m,
        n,
        left_indices,
        right_indices,
    ))
}

pub(crate) fn unfold_split_inner(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<UnfoldSplitInnerResult> {
    let rank = t.indices.len();

    // Validate rank
    anyhow::ensure!(rank >= 2, "Tensor must have rank >= 2, got rank {}", rank);

    let left_len = left_inds.len();

    // Validate split: must be a proper subset
    anyhow::ensure!(
        left_len > 0 && left_len < rank,
        "Left indices must be a non-empty proper subset of tensor indices (0 < left_len < rank), got left_len={}, rank={}",
        left_len,
        rank
    );

    // Validate that all left_inds are in the tensor and there are no duplicates
    let tensor_set: HashSet<_> = t.indices.iter().collect();
    let mut left_set = HashSet::new();

    for left_idx in left_inds {
        anyhow::ensure!(
            tensor_set.contains(left_idx),
            "Index in left_inds not found in tensor"
        );
        anyhow::ensure!(left_set.insert(left_idx), "Duplicate index in left_inds");
    }

    // Build right_inds: all indices not in left_inds, in original order
    let mut right_inds = Vec::new();
    for idx in &t.indices {
        if !left_set.contains(idx) {
            right_inds.push(idx.clone());
        }
    }

    // Build new_indices: left_inds first, then right_inds
    let mut new_indices = Vec::with_capacity(rank);
    new_indices.extend_from_slice(left_inds);
    new_indices.extend_from_slice(&right_inds);

    // Permute tensor to have left indices first, then right indices
    let unfolded = t.permute_indices(&new_indices)?;

    // Compute matrix dimensions
    let unfolded_dims = unfolded.dims();
    let m: usize = unfolded_dims[..left_len].iter().product();
    let n: usize = unfolded_dims[left_len..].iter().product();

    let matrix_tensor = unfolded.try_materialized_inner()?.reshape(&[m, n])?;

    Ok((
        matrix_tensor,
        left_len,
        m,
        n,
        left_inds.to_vec(),
        right_inds,
    ))
}

// ============================================================================
// TensorIndex implementation for TensorDynLen
// ============================================================================

use crate::tensor_index::TensorIndex;

impl TensorIndex for TensorDynLen {
    type Index = DynIndex;
    type Error = TensorDynLenError;

    fn external_indices(&self) -> Vec<DynIndex> {
        // For TensorDynLen, all indices are external.
        self.indices.clone()
    }

    fn num_external_indices(&self) -> usize {
        self.indices.len()
    }

    fn replaceind(
        &self,
        old_index: &DynIndex,
        new_index: &DynIndex,
    ) -> std::result::Result<Self, Self::Error> {
        // Delegate to the inherent method.
        TensorDynLen::replaceind(self, old_index, new_index).map_err(Self::Error::from)
    }

    fn replaceinds(
        &self,
        old_indices: &[DynIndex],
        new_indices: &[DynIndex],
    ) -> std::result::Result<Self, Self::Error> {
        // Delegate to the inherent method.
        TensorDynLen::replaceinds(self, old_indices, new_indices).map_err(Self::Error::from)
    }
}

// ============================================================================
// TensorLike implementation for TensorDynLen
// ============================================================================

use crate::tensor_like::{
    FactorizeError, FactorizeOptions, FactorizeResult, TensorConstructionLike,
    TensorContractionLike, TensorFactorizationLike, TensorVectorSpace,
};

impl TensorVectorSpace for TensorDynLen {
    fn norm_squared(&self) -> std::result::Result<f64, Self::Error> {
        TensorDynLen::norm_squared(self)
    }

    fn maxabs(&self) -> std::result::Result<f64, Self::Error> {
        TensorDynLen::maxabs(self)
    }

    fn isapprox(
        &self,
        other: &Self,
        atol: f64,
        rtol: f64,
    ) -> std::result::Result<bool, Self::Error> {
        TensorDynLen::isapprox(self, other, atol, rtol)
    }

    fn axpby(
        &self,
        a: crate::AnyScalar,
        other: &Self,
        b: crate::AnyScalar,
    ) -> std::result::Result<Self, Self::Error> {
        TensorDynLen::axpby(self, a, other, b).map_err(Self::Error::from)
    }

    fn scale(&self, scalar: crate::AnyScalar) -> std::result::Result<Self, Self::Error> {
        TensorDynLen::scale(self, scalar).map_err(Self::Error::from)
    }

    fn inner_product(&self, other: &Self) -> std::result::Result<crate::AnyScalar, Self::Error> {
        TensorDynLen::inner_product(self, other).map_err(Self::Error::from)
    }
}

impl TensorFactorizationLike for TensorDynLen {
    fn factorize(
        &self,
        left_inds: &[DynIndex],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        crate::factorize::factorize(self, left_inds, options)
    }

    fn factorize_full_rank(
        &self,
        left_inds: &[DynIndex],
        alg: crate::FactorizeAlg,
        canonical: crate::Canonical,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        crate::factorize::factorize_full_rank(self, left_inds, alg, canonical)
    }
}

impl TensorContractionLike for TensorDynLen {
    fn conj(&self) -> Self {
        // Delegate to the inherent method (complex conjugate for dense tensors)
        TensorDynLen::conj(self)
    }

    fn direct_sum(
        &self,
        other: &Self,
        pairs: &[(DynIndex, DynIndex)],
    ) -> std::result::Result<crate::tensor_like::DirectSumResult<Self>, Self::Error> {
        let (tensor, new_indices) =
            crate::direct_sum::direct_sum(self, other, pairs).map_err(Self::Error::from)?;
        Ok(crate::tensor_like::DirectSumResult {
            tensor,
            new_indices,
        })
    }

    fn outer_product(&self, other: &Self) -> std::result::Result<Self, Self::Error> {
        super::contract::outer_product(self, other).map_err(Self::Error::from)
    }

    fn permuteinds(&self, new_order: &[DynIndex]) -> std::result::Result<Self, Self::Error> {
        // Delegate to the inherent method
        TensorDynLen::permute_indices(self, new_order).map_err(Self::Error::from)
    }

    fn fuse_indices(
        &self,
        old_indices: &[DynIndex],
        new_index: DynIndex,
        order: LinearizationOrder,
    ) -> std::result::Result<Self, Self::Error> {
        TensorDynLen::fuse_indices(self, old_indices, new_index, order).map_err(Self::Error::from)
    }

    fn contract(tensors: &[&Self]) -> std::result::Result<Self, Self::Error> {
        super::contract::contract(tensors).map_err(Self::Error::from)
    }

    fn contract_pair(&self, other: &Self) -> std::result::Result<Self, Self::Error> {
        super::contract::contract_pair(self, other).map_err(Self::Error::from)
    }
}

impl TensorConstructionLike for TensorDynLen {
    fn select_indices(
        &self,
        selected_indices: &[DynIndex],
        positions: &[usize],
    ) -> std::result::Result<Self, Self::Error> {
        TensorDynLen::select_indices(self, selected_indices, positions).map_err(Self::Error::from)
    }

    fn diagonal(
        input_index: &DynIndex,
        output_index: &DynIndex,
    ) -> std::result::Result<Self, Self::Error> {
        let dim = input_index.dim();
        if dim != output_index.dim() {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: input index has dim {}, output has dim {}",
                dim,
                output_index.dim(),
            )
            .into());
        }

        TensorDynLen::from_diag(
            vec![input_index.clone(), output_index.clone()],
            vec![1.0_f64; dim],
        )
        .map_err(Self::Error::from)
    }

    fn scalar_one() -> std::result::Result<Self, Self::Error> {
        TensorDynLen::from_dense(vec![], vec![1.0_f64]).map_err(Self::Error::from)
    }

    fn ones(indices: &[DynIndex]) -> std::result::Result<Self, Self::Error> {
        if indices.is_empty() {
            return <Self as TensorConstructionLike>::scalar_one();
        }
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
        let total_size = checked_total_size(&dims).map_err(Self::Error::from)?;
        TensorDynLen::from_dense(indices.to_vec(), vec![1.0_f64; total_size])
            .map_err(Self::Error::from)
    }

    fn onehot(index_vals: &[(DynIndex, usize)]) -> std::result::Result<Self, Self::Error> {
        if index_vals.is_empty() {
            return <Self as TensorConstructionLike>::scalar_one();
        }
        let indices: Vec<DynIndex> = index_vals.iter().map(|(idx, _)| idx.clone()).collect();
        let vals: Vec<usize> = index_vals.iter().map(|(_, v)| *v).collect();
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();

        for (k, (&v, &d)) in vals.iter().zip(dims.iter()).enumerate() {
            if v >= d {
                return Err(anyhow::anyhow!(
                    "onehot: value {} at position {} is >= dimension {}",
                    v,
                    k,
                    d
                )
                .into());
            }
        }

        let total_size = checked_total_size(&dims).map_err(Self::Error::from)?;
        let mut data = vec![0.0_f64; total_size];

        let offset = column_major_offset(&dims, &vals).map_err(Self::Error::from)?;
        data[offset] = 1.0;

        Self::from_dense(indices, data).map_err(Self::Error::from)
    }

    // delta() uses the default implementation via diagonal() and outer_product()
}

fn checked_total_size(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |acc, &d| {
        if d == 0 {
            return Err(anyhow::anyhow!("invalid dimension 0"));
        }
        acc.checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))
    })
}

fn column_major_offset(dims: &[usize], vals: &[usize]) -> Result<usize> {
    if dims.len() != vals.len() {
        return Err(anyhow::anyhow!(
            "column_major_offset: dims.len() != vals.len()"
        ));
    }
    checked_total_size(dims)?;

    let mut offset = 0usize;
    let mut stride = 1usize;
    for (k, (&v, &d)) in vals.iter().zip(dims.iter()).enumerate() {
        if d == 0 {
            return Err(anyhow::anyhow!("invalid dimension 0 at position {}", k));
        }
        if v >= d {
            return Err(anyhow::anyhow!(
                "column_major_offset: value {} at position {} is >= dimension {}",
                v,
                k,
                d
            ));
        }
        let term = v
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("column_major_offset: overflow"))?;
        offset = offset
            .checked_add(term)
            .ok_or_else(|| anyhow::anyhow!("column_major_offset: overflow"))?;
        stride = stride
            .checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("column_major_offset: overflow"))?;
    }
    Ok(offset)
}

// ============================================================================
// High-level API for tensor construction (avoids direct Storage access)
// ============================================================================

impl TensorDynLen {
    fn any_scalar_payload_to_complex(data: Vec<AnyScalar>) -> Vec<Complex64> {
        data.into_iter()
            .map(|value| {
                value
                    .as_c64()
                    .unwrap_or_else(|| Complex64::new(value.real(), 0.0))
            })
            .collect()
    }

    fn any_scalar_payload_to_real(data: Vec<AnyScalar>) -> Vec<f64> {
        data.into_iter().map(|value| value.real()).collect()
    }

    fn validate_dense_payload_len(data_len: usize, dims: &[usize]) -> Result<()> {
        let expected_len = checked_total_size(dims)?;
        anyhow::ensure!(
            data_len == expected_len,
            "dense payload length {} does not match dims {:?} (expected {})",
            data_len,
            dims,
            expected_len
        );
        Ok(())
    }

    fn validate_diag_payload_len(data_len: usize, dims: &[usize]) -> Result<()> {
        anyhow::ensure!(
            !dims.is_empty(),
            "diagonal tensor construction requires at least one index"
        );
        Self::validate_diag_dims(dims)?;
        anyhow::ensure!(
            data_len == dims[0],
            "diagonal payload length {} does not match diagonal dimension {}",
            data_len,
            dims[0]
        );
        Ok(())
    }

    /// Create a tensor from dense data with explicit indices.
    ///
    /// This is the recommended high-level API for creating tensors from raw data.
    /// It avoids direct access to `Storage` internals.
    ///
    /// # Type Parameters
    /// * `T` - Scalar type (`f32`, `f64`, `Complex32`, or `Complex64`)
    ///
    /// # Arguments
    /// * `indices` - Vector of indices for the tensor
    /// * `data` - Tensor data in column-major order
    ///
    /// # Errors
    /// Returns an error when the data length does not match the index dimension
    /// product (a dimension mismatch).
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i, j], data).unwrap();
    /// assert_eq!(tensor.dims(), vec![2, 3]);
    /// ```
    pub fn from_dense<T: TensorElement>(indices: Vec<DynIndex>, data: Vec<T>) -> Result<Self> {
        let dims = Self::expected_dims_from_indices(&indices);
        Self::validate_indices(&indices)?;
        Self::validate_dense_payload_len(data.len(), &dims)?;
        let native = dense_native_tensor_from_col_major(&data, &dims)?;
        Self::from_native(indices, native)
    }

    /// Create a tensor from dense payload data provided as [`AnyScalar`] values.
    ///
    /// This is the preferred public API when the caller only knows the scalar
    /// type at runtime.
    ///
    /// # Errors
    /// Returns an error when the payload length does not match the index dimension
    /// product (a dimension mismatch) or a scalar conversion fails.
    /// # Examples
    /// ```
    /// use tensor4all_core::{AnyScalar, TensorDynLen};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense_any(
    ///     vec![i, j],
    ///     vec![
    ///         AnyScalar::new_real(1.0),
    ///         AnyScalar::new_complex(0.0, 1.0),
    ///         AnyScalar::new_real(2.0),
    ///         AnyScalar::new_real(3.0),
    ///     ],
    /// ).unwrap();
    ///
    /// assert!(tensor.is_complex());
    /// assert_eq!(tensor.dims(), vec![2, 2]);
    /// ```
    pub fn from_dense_any(indices: Vec<DynIndex>, data: Vec<AnyScalar>) -> Result<Self> {
        if data.iter().any(AnyScalar::is_complex) {
            Self::from_dense(indices, Self::any_scalar_payload_to_complex(data))
        } else {
            Self::from_dense(indices, Self::any_scalar_payload_to_real(data))
        }
    }

    /// Create a diagonal tensor from diagonal payload data with explicit indices.
    ///
    /// All indices must have the same dimension, and `data.len()` must equal
    /// that dimension. The resulting tensor has nonzero entries only on
    /// the multi-index diagonal (`T[i,i,...,i] = data[i]`).
    ///
    /// The returned tensor preserves diagonal metadata; use
    /// [`TensorDynLen::is_diag`] or [`TensorDynLen::storage_kind`] to inspect
    /// that representation. `f32` and `Complex32` values remain eager and are
    /// never promoted into compact `f64`/`Complex64` storage.
    ///
    /// # Errors
    /// Returns an error when the index dimensions are unequal or the payload
    /// length does not match the diagonal dimension (a dimension mismatch).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let j = DynIndex::new_dyn(3);
    /// let diag = TensorDynLen::from_diag(vec![i, j], vec![1.0, 2.0, 3.0]).unwrap();
    /// assert!(diag.is_diag());
    ///
    /// let data = diag.to_vec::<f64>().unwrap();
    /// // 3x3 identity-like: [1,0,0, 0,2,0, 0,0,3] in column-major
    /// assert!((data[0] - 1.0).abs() < 1e-12);
    /// assert!((data[4] - 2.0).abs() < 1e-12);
    /// assert!((data[8] - 3.0).abs() < 1e-12);
    /// assert!((data[1]).abs() < 1e-12);  // off-diagonal is zero
    /// ```
    pub fn from_diag<T: TensorElement>(indices: Vec<DynIndex>, data: Vec<T>) -> Result<Self> {
        let dims = Self::expected_dims_from_indices(&indices);
        Self::validate_indices(&indices)?;
        Self::validate_diag_payload_len(data.len(), &dims)?;
        let native = diag_native_tensor_from_col_major(&data, dims.len())?;
        Self::from_native_with_axis_classes(indices, native, Self::diag_axis_classes(dims.len()))
    }

    /// Create a diagonal tensor from diagonal payload data provided as
    /// [`AnyScalar`] values.
    ///
    /// This is the preferred public API when the caller only knows the scalar
    /// type at runtime.
    ///
    /// # Errors
    /// Returns an error when the index dimensions are unequal or the payload
    /// length does not match (a dimension mismatch), or a scalar conversion
    /// fails.
    /// # Examples
    /// ```
    /// use tensor4all_core::{AnyScalar, TensorDynLen};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_diag_any(
    ///     vec![i, j],
    ///     vec![AnyScalar::new_real(1.0), AnyScalar::new_complex(2.0, -1.0)],
    /// ).unwrap();
    ///
    /// assert!(tensor.is_complex());
    /// assert_eq!(tensor.dims(), vec![2, 2]);
    /// ```
    pub fn from_diag_any(indices: Vec<DynIndex>, data: Vec<AnyScalar>) -> Result<Self> {
        if data.iter().any(AnyScalar::is_complex) {
            Self::from_diag(indices, Self::any_scalar_payload_to_complex(data))
        } else {
            Self::from_diag(indices, Self::any_scalar_payload_to_real(data))
        }
    }

    /// Create a copy tensor whose nonzero entries are `value` on the diagonal.
    ///
    /// For indices `[i, j, k]`, the returned tensor satisfies
    /// `T[i, j, k] = value` when `i = j = k`, and zero otherwise.
    ///
    /// # Errors
    /// Returns an error when the index dimensions are unequal (a dimension
    /// mismatch) or the construction fails.
    /// # Examples
    /// ```
    /// use tensor4all_core::{AnyScalar, TensorDynLen};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(2);
    /// let k = Index::new_dyn(2);
    /// let tensor = TensorDynLen::copy_tensor(
    ///     vec![i, j, k],
    ///     AnyScalar::new_real(1.0),
    /// ).unwrap();
    ///
    /// assert_eq!(tensor.dims(), vec![2, 2, 2]);
    /// ```
    pub fn copy_tensor(indices: Vec<DynIndex>, value: AnyScalar) -> Result<Self> {
        if indices.is_empty() {
            return Self::from_dense_any(vec![], vec![value]);
        }
        let dim = indices[0].dim();
        let data = vec![value; dim];
        Self::from_diag_any(indices, data)
    }

    /// Replace multiple tensor indices with one fused index using an exact local reshape.
    ///
    /// The indices in `old_indices` identify the axes to fuse by ID and also
    /// define the coordinate order used inside `new_index`. The new fused index
    /// is inserted at the earliest axis position among the fused axes; all
    /// other axes keep their original relative order. Use
    /// [`LinearizationOrder::ColumnMajor`] to match tensor4all's dense vector
    /// layout, or [`LinearizationOrder::RowMajor`] when interoperating with
    /// row-major fused coordinates.
    ///
    /// # Arguments
    /// * `old_indices` - Non-empty list of existing tensor indices to replace.
    ///   Each index is matched by ID, must appear exactly once in the tensor,
    ///   must have the same dimension as the matched tensor axis, and must not
    ///   be duplicated in this list.
    /// * `new_index` - Replacement index whose dimension must equal the product
    ///   of the dimensions in `old_indices`.
    /// * `order` - Linearization convention used to encode the old coordinates
    ///   into the single coordinate of `new_index`.
    ///
    /// # Returns
    /// A tensor with the same element type and values, but with `old_indices`
    /// replaced by `new_index`.
    ///
    /// # Errors
    /// Returns an error if `old_indices` is empty, contains duplicate IDs,
    /// references an index not present in the tensor, if the fused dimension
    /// does not match the product of the old dimensions, if the replacement
    /// would duplicate a kept index, or if the dense reshape cannot be
    /// represented without overflow.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, LinearizationOrder, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let fused = DynIndex::new_link(4).unwrap();
    /// let tensor = TensorDynLen::from_dense(
    ///     vec![i.clone(), j.clone()],
    ///     vec![1.0, 2.0, 3.0, 4.0],
    /// ).unwrap();
    ///
    /// let fused_tensor = tensor
    ///     .fuse_indices(&[i.clone(), j.clone()], fused.clone(), LinearizationOrder::ColumnMajor)
    ///     .unwrap();
    /// assert_eq!(fused_tensor.dims(), vec![4]);
    ///
    /// let roundtrip = fused_tensor
    ///     .unfuse_index(&fused, &[i, j], LinearizationOrder::ColumnMajor)
    ///     .unwrap();
    /// assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0).unwrap());
    /// ```
    pub fn fuse_indices(
        &self,
        old_indices: &[DynIndex],
        new_index: DynIndex,
        order: LinearizationOrder,
    ) -> Result<Self> {
        anyhow::ensure!(
            !old_indices.is_empty(),
            "fuse_indices requires at least one index to fuse"
        );

        let old_dims = self.dims();
        let mut seen_indices = HashSet::new();
        let mut old_axes = Vec::with_capacity(old_indices.len());
        for old_index in old_indices {
            anyhow::ensure!(
                seen_indices.insert(old_index),
                "duplicate index in old_indices"
            );
            let axis = self
                .indices
                .iter()
                .position(|idx| idx == old_index)
                .ok_or_else(|| anyhow::anyhow!("index {:?} not found in tensor", old_index))?;
            anyhow::ensure!(
                old_index.dim() == old_dims[axis],
                "old index dimension does not match tensor axis dimension"
            );
            old_axes.push(axis);
        }

        let fused_dims: Vec<usize> = old_axes.iter().map(|&axis| old_dims[axis]).collect();
        let fused_product = checked_product(&fused_dims)?;
        anyhow::ensure!(
            fused_product == new_index.dim(),
            "product of old index dimensions must match the replacement index dimension"
        );

        let insertion_axis =
            old_axes.iter().copied().min().ok_or_else(|| {
                anyhow::anyhow!("fuse_indices requires at least one index to fuse")
            })?;
        let old_axis_set: HashSet<usize> = old_axes.iter().copied().collect();

        let mut result_indices =
            Vec::with_capacity(self.indices.len() - old_indices.len() + 1usize);
        for (axis, index) in self.indices.iter().enumerate() {
            if axis == insertion_axis {
                result_indices.push(new_index.clone());
            }
            if !old_axis_set.contains(&axis) {
                result_indices.push(index.clone());
            }
        }
        let mut result_seen = HashSet::new();
        for index in &result_indices {
            anyhow::ensure!(
                result_seen.insert(index),
                "fuse_indices result would contain duplicate index"
            );
        }
        Self::validate_indices(&result_indices)?;

        let mut new_dims = Vec::with_capacity(old_dims.len() - old_indices.len() + 1usize);
        for (axis, dim) in old_dims.iter().copied().enumerate() {
            if axis == insertion_axis {
                new_dims.push(new_index.dim());
            }
            if !old_axis_set.contains(&axis) {
                new_dims.push(dim);
            }
        }

        self.ensure_shape_packing_preserves_ad("fuse_indices")?;

        let mut grouped_axes = old_axes.clone();
        if matches!(order, LinearizationOrder::RowMajor) {
            grouped_axes.reverse();
        }
        let mut perm = Vec::with_capacity(self.indices.len());
        perm.extend((0..insertion_axis).filter(|axis| !old_axis_set.contains(axis)));
        perm.extend(grouped_axes);
        perm.extend(
            ((insertion_axis + 1)..self.indices.len()).filter(|axis| !old_axis_set.contains(axis)),
        );
        debug_assert_eq!(perm.len(), self.indices.len());

        let packed = self.permute(&perm)?;
        let reshaped = packed.try_materialized_inner()?.reshape(&new_dims)?;
        Self::from_inner(result_indices, reshaped)
    }

    /// Replace one fused index with multiple indices using an exact reshape.
    ///
    /// The caller must specify how the old fused index should be decoded into
    /// the new indices via `order`.
    ///
    /// # Errors
    /// Returns an error when the fused dimension does not equal the product of
    /// the new index dimensions (a dimension mismatch).
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, LinearizationOrder, TensorDynLen};
    ///
    /// let fused = DynIndex::new_dyn(4);
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    ///
    /// let unfused = tensor
    ///     .unfuse_index(&fused, &[i.clone(), j.clone()], LinearizationOrder::ColumnMajor)
    ///     .unwrap();
    ///
    /// let expected = TensorDynLen::from_dense(vec![i, j], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert!(unfused.isapprox(&expected, 1e-12, 0.0).unwrap());
    /// ```
    pub fn unfuse_index(
        &self,
        old_index: &DynIndex,
        new_indices: &[DynIndex],
        order: LinearizationOrder,
    ) -> Result<Self> {
        anyhow::ensure!(
            !new_indices.is_empty(),
            "unfuse_index requires at least one replacement index"
        );

        let axis = self
            .indices
            .iter()
            .position(|idx| idx == old_index)
            .ok_or_else(|| anyhow::anyhow!("index {:?} not found in tensor", old_index))?;

        let replacement_dims: Vec<usize> = new_indices.iter().map(DynIndex::dim).collect();
        let replacement_product = checked_product(&replacement_dims)?;
        anyhow::ensure!(
            replacement_product == old_index.dim(),
            "product of new index dimensions must match the replaced index dimension"
        );

        let mut result_indices =
            Vec::with_capacity(self.indices.len() - 1usize + new_indices.len());
        result_indices.extend_from_slice(&self.indices[..axis]);
        result_indices.extend(new_indices.iter().cloned());
        result_indices.extend_from_slice(&self.indices[axis + 1..]);
        Self::validate_indices(&result_indices)?;

        let old_dims = self.dims();
        let mut new_dims = Vec::with_capacity(old_dims.len() - 1usize + replacement_dims.len());
        new_dims.extend_from_slice(&old_dims[..axis]);
        new_dims.extend_from_slice(&replacement_dims);
        new_dims.extend_from_slice(&old_dims[axis + 1..]);

        self.ensure_shape_packing_preserves_ad("unfuse_index")?;

        let mut grouped_indices = new_indices.to_vec();
        let mut grouped_dims = replacement_dims.clone();
        if matches!(order, LinearizationOrder::RowMajor) {
            grouped_indices.reverse();
            grouped_dims.reverse();
        }
        let mut packed_indices =
            Vec::with_capacity(self.indices.len() - 1usize + grouped_indices.len());
        packed_indices.extend_from_slice(&self.indices[..axis]);
        packed_indices.extend(grouped_indices);
        packed_indices.extend_from_slice(&self.indices[axis + 1..]);

        let mut packed_dims = Vec::with_capacity(old_dims.len() - 1usize + grouped_dims.len());
        packed_dims.extend_from_slice(&old_dims[..axis]);
        packed_dims.extend_from_slice(&grouped_dims);
        packed_dims.extend_from_slice(&old_dims[axis + 1..]);

        let reshaped = self.try_materialized_inner()?.reshape(&packed_dims)?;
        let packed = Self::from_inner(packed_indices, reshaped)?;
        if matches!(order, LinearizationOrder::ColumnMajor) {
            Ok(packed)
        } else {
            packed.permute_indices(&result_indices)
        }
    }

    /// Create a scalar (0-dimensional) tensor from a supported element value.
    ///
    /// # Errors
    /// Returns an error when the element type is not supported (an
    /// unsupported-dtype failure).
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    ///
    /// let scalar = TensorDynLen::scalar(42.0).unwrap();
    /// assert_eq!(scalar.dims(), Vec::<usize>::new());
    /// assert_eq!(scalar.only().unwrap().real(), 42.0);
    /// ```
    pub fn scalar<T: TensorElement>(value: T) -> Result<Self> {
        Self::from_dense(vec![], vec![value])
    }

    /// Create a tensor filled with zeros of a supported element type.
    ///
    /// # Errors
    /// Returns an error when the dimension product overflows (an overflow failure)
    /// or the element type is unsupported.
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor = TensorDynLen::zeros::<f64>(vec![i, j]).unwrap();
    /// assert_eq!(tensor.dims(), vec![2, 3]);
    /// ```
    pub fn zeros<T: TensorElement + Zero + Clone>(indices: Vec<DynIndex>) -> Result<Self> {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
        let size: usize = dims.iter().product();
        Self::from_dense(indices, vec![T::zero(); size])
    }
}

// ============================================================================
// High-level API for data extraction (avoids direct .storage() access)
// ============================================================================

impl TensorDynLen {
    /// Extract tensor data as a column-major `Vec<T>`.
    ///
    /// # Type Parameters
    /// * `T` - The scalar element type (`f32`, `f64`, `Complex32`, or
    ///   `Complex64`).
    ///
    /// # Returns
    /// A vector of the tensor data in column-major order.
    ///
    /// # Errors
    /// Returns an error when the tensor dtype does not match the requested element
    /// type (a scalar-kind mismatch) or materialization fails.
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// let data = tensor.to_vec::<f64>().unwrap();
    /// assert_eq!(data, &[1.0, 2.0]);
    /// ```
    pub fn to_vec<T: TensorElement>(&self) -> Result<Vec<T>> {
        native_tensor_primal_to_dense_col_major(self.as_native()?)
    }

    /// Consume the tensor and return its indices with dense column-major values.
    ///
    /// Use this when a caller needs to move index metadata and dense payload
    /// values across an API boundary. The returned values are ordered with the
    /// first tensor index varying fastest. Compact diagonal or structured
    /// storage is materialized into dense logical values.
    ///
    /// # Type Parameters
    /// * `T` - The scalar element type to extract: `f32`, `f64`, `Complex32`,
    ///   or `Complex64`.
    ///
    /// # Returns
    /// The tensor's original indices and dense column-major flat data.
    ///
    /// # Errors
    /// Returns an error when the tensor dtype does not match the requested element
    /// type (a scalar-kind mismatch) or materialization fails.
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(
    ///     vec![i.clone(), j.clone()],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0],
    /// ).unwrap();
    ///
    /// let (indices, data) = tensor.into_dense_col_major_parts::<f64>().unwrap();
    ///
    /// assert_eq!(indices, vec![i, j]);
    /// assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn into_dense_col_major_parts<T: TensorElement>(self) -> Result<(Vec<DynIndex>, Vec<T>)> {
        anyhow::ensure!(
            !self.tracks_grad(),
            "TensorDynLen::into_dense_col_major_parts cannot consume tensors with tracked autodiff state"
        );
        let data = self.to_vec::<T>()?;
        Ok((self.indices, data))
    }

    /// Extract tensor data as a column-major `Vec<f64>`.
    ///
    /// Prefer the generic [`to_vec::<f64>()`](Self::to_vec) method.
    /// This wrapper is kept for C API compatibility.
    /// # Errors
    /// Returns an error when the tensor is not f64-compatible (a dtype mismatch)
    /// or materialization fails.
    ///
    pub fn as_slice_f64(&self) -> Result<Vec<f64>> {
        self.to_vec::<f64>()
    }

    /// Extract tensor data as a column-major `Vec<Complex64>`.
    ///
    /// Prefer the generic [`to_vec::<Complex64>()`](Self::to_vec) method.
    /// This wrapper is kept for C API compatibility.
    /// # Errors
    /// Returns an error when the tensor is not c64-compatible (a dtype mismatch)
    /// or materialization fails.
    ///
    pub fn as_slice_c64(&self) -> Result<Vec<Complex64>> {
        self.to_vec::<Complex64>()
    }

    /// Check if the tensor has `f64` storage.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// assert!(tensor.is_f64());
    /// assert!(!tensor.is_complex());
    /// ```
    pub fn is_f64(&self) -> bool {
        self.storage.dtype() == Some(DType::F64)
    }

    /// Check if the tensor has `f32` storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let tensor = TensorDynLen::from_dense(
    ///     vec![DynIndex::new_dyn(2)],
    ///     vec![1.0_f32, 2.0],
    /// )
    /// .unwrap();
    /// assert!(tensor.is_f32());
    /// ```
    pub fn is_f32(&self) -> bool {
        self.storage.dtype() == Some(DType::F32)
    }

    /// Check if the tensor has complex-32 storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let tensor = TensorDynLen::from_dense(
    ///     vec![DynIndex::new_dyn(2)],
    ///     vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)],
    /// )
    /// .unwrap();
    /// assert!(tensor.is_c32());
    /// ```
    pub fn is_c32(&self) -> bool {
        self.storage.dtype() == Some(DType::C32)
    }

    /// Check if the tensor has complex-64 storage.
    ///
    /// # Example
    /// ```
    /// use num_complex::Complex64;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(
    ///     vec![i],
    ///     vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    /// )
    /// .unwrap();
    /// assert!(tensor.is_c64());
    /// ```
    pub fn is_c64(&self) -> bool {
        self.storage.is_c64()
    }

    /// Check whether the tensor carries diagonal logical axis metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_tensorbackend::Storage;
    ///
    /// // Tensors from `from_dense` use dense storage
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let dense = TensorDynLen::from_dense(vec![i, j], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    /// assert!(!dense.is_diag());
    ///
    /// // Diagonal metadata is preserved when constructing from diagonal storage.
    /// let k = DynIndex::new_dyn(2);
    /// let l = DynIndex::new_dyn(2);
    /// let diag = TensorDynLen::from_storage(
    ///     vec![k, l],
    ///     Storage::from_diag_col_major(vec![1.0, 2.0], 2)
    ///         .map(std::sync::Arc::new)
    ///         .unwrap(),
    /// )
    /// .unwrap();
    /// assert!(diag.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        self.storage.is_diag()
    }

    /// Check if the tensor has complex storage (C64).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use num_complex::Complex64;
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let real_t = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// assert!(!real_t.is_complex());
    ///
    /// let complex_t = TensorDynLen::from_dense(
    ///     vec![i],
    ///     vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    /// ).unwrap();
    /// assert!(complex_t.is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        self.storage.is_complex()
    }
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("dimension product overflow"))
    })
}

fn increment_col_major_coordinate(coords: &mut [usize], dims: &[usize]) {
    let mut carry = true;
    for (coordinate, &dim) in coords.iter_mut().zip(dims.iter()) {
        if !carry {
            break;
        }
        *coordinate += 1;
        if *coordinate == dim {
            *coordinate = 0;
        } else {
            carry = false;
        }
    }
}

fn map_payload_support_coordinate(
    source_axis_classes: &[usize],
    target_axis_classes: &[usize],
    source_to_target_axes: &[usize],
    source_coords: &[usize],
    target_coords: &mut [usize],
    target_seen: &mut [bool],
) -> bool {
    if source_axis_classes.len() != source_to_target_axes.len()
        || target_coords.len() != target_seen.len()
    {
        return false;
    }
    target_seen.fill(false);
    for (source_axis, &target_axis) in source_to_target_axes.iter().enumerate() {
        let Some(&source_class) = source_axis_classes.get(source_axis) else {
            return false;
        };
        let Some(&target_class) = target_axis_classes.get(target_axis) else {
            return false;
        };
        let Some(&source_value) = source_coords.get(source_class) else {
            return false;
        };
        let Some(target_value) = target_coords.get_mut(target_class) else {
            return false;
        };
        if target_seen[target_class] {
            if *target_value != source_value {
                return false;
            }
        } else {
            *target_value = source_value;
            target_seen[target_class] = true;
        }
    }
    target_seen.iter().all(|&seen| seen)
}

fn decode_col_major_linear(linear: usize, dims: &[usize]) -> Result<Vec<usize>> {
    let total = checked_product(dims)?;
    anyhow::ensure!(
        linear < total,
        "linear offset {} out of bounds for dims {:?}",
        linear,
        dims
    );
    let mut remaining = linear;
    let mut out = Vec::with_capacity(dims.len());
    for &dim in dims {
        out.push(remaining % dim);
        remaining /= dim;
    }
    Ok(out)
}

fn encode_col_major_linear(indices: &[usize], dims: &[usize]) -> Result<usize> {
    anyhow::ensure!(
        indices.len() == dims.len(),
        "index rank {} does not match dims {:?}",
        indices.len(),
        dims
    );
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (&index, &dim) in indices.iter().zip(dims.iter()) {
        anyhow::ensure!(
            index < dim,
            "index {} out of bounds for dimension {}",
            index,
            dim
        );
        linear += index * stride;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("stride overflow"))?;
    }
    Ok(linear)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};
    use std::cell::Cell;
    use tensor4all_tensorbackend::StorageError;

    #[test]
    fn structured_contraction_does_not_install_logical_dense_cache() {
        let n = 8;
        let left = DynIndex::new_dyn(n);
        let site = DynIndex::new_dyn(3);
        let right = DynIndex::new_dyn(n);
        let far = DynIndex::new_dyn(n);
        let end = DynIndex::new_dyn(n);
        let a = TensorDynLen::from_copy_selector(left, site.clone(), right.clone(), 1, 1.0_f64)
            .unwrap();
        let b =
            TensorDynLen::from_copy_selector(right, site.clone(), far.clone(), 1, 2.0_f64).unwrap();
        let c = TensorDynLen::from_copy_selector(far, site.clone(), end, 1, 3.0_f64).unwrap();
        let result = crate::defaults::contract::contract_with_options(
            &[&a, &b, &c],
            crate::defaults::contract::ContractionOptions::new()
                .with_retain_indices(std::slice::from_ref(&site)),
        )
        .unwrap();

        assert_eq!(result.storage_kind(), StorageKind::Structured);
        assert!(result.eager_cache.get().is_none());
        assert_eq!(result.storage().unwrap().payload_len(), n * 3);
    }

    #[test]
    fn structured_metrics_use_authoritative_compact_payload_for_all_dtypes() {
        fn check(tensor: TensorDynLen, expected_sum: f64, expected_norm_squared: f64) {
            assert!(matches!(tensor.storage, TensorDynLenStorage::Compact(_)));
            assert!(tensor.eager_cache.get().is_none());
            assert!((tensor.sum().unwrap().real() - expected_sum).abs() < 1.0e-6);
            assert!((tensor.norm_squared().unwrap() - expected_norm_squared).abs() < 1.0e-6);
            assert!((tensor.maxabs().unwrap() - 2.0).abs() < 1.0e-6);
            assert!(tensor.isapprox(&tensor, 0.0, 0.0).unwrap());
            assert!(tensor.eager_cache.get().is_none());
        }

        let indices = || vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
        check(
            TensorDynLen::from_diag(indices(), vec![1.0_f32, 2.0]).unwrap(),
            3.0,
            5.0,
        );
        check(
            TensorDynLen::from_diag(indices(), vec![1.0_f64, 2.0]).unwrap(),
            3.0,
            5.0,
        );
        check(
            TensorDynLen::from_diag(
                indices(),
                vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
            )
            .unwrap(),
            3.0,
            5.0,
        );
        check(
            TensorDynLen::from_diag(
                indices(),
                vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            )
            .unwrap(),
            3.0,
            5.0,
        );
    }

    #[test]
    fn payload_storage_error_retains_typed_source() {
        let storage = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
        let error = storage.scalar_at(&[2]).unwrap_err();
        assert!(matches!(error, StorageError::InvalidStructuredStorage(_)));
    }

    #[test]
    fn materialization_error_retains_backend_source() {
        let native = NativeTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
        let inner = EagerTensor::from_tensor_in(native, default_eager_ctx().unwrap());
        let storage = TensorDynLenStorage::Eager {
            inner: Arc::new(inner),
            axis_classes: vec![0, 0],
        };

        let error = storage.materialize(2).unwrap_err();
        assert!(matches!(error, TensorStorageError::Materialization { .. }));
        assert!(std::error::Error::source(&error).is_some());
    }

    fn conjugate_with_injected_failure(
        tensor: &TensorDynLen,
        target: *const EagerTensor,
        message: &'static str,
    ) -> TensorDynLen {
        let calls = Cell::new(0usize);
        let conjugated = tensor.conj_with(&|inner| {
            calls.set(calls.get() + 1);
            if std::ptr::eq(inner, target) {
                Err(Arc::new(std::io::Error::other(message)) as _)
            } else {
                conjugate_eager(inner)
            }
        });
        assert!(calls.get() > 0, "injected closure was not reached");
        conjugated
    }

    fn assert_unwrapped_conjugation_error(
        tensor: TensorDynLen,
        target: *const EagerTensor,
        message: &'static str,
    ) -> TensorDynLen {
        let conjugated = conjugate_with_injected_failure(&tensor, target, message);
        let error = conjugated.to_storage().unwrap_err();
        assert!(matches!(error, TensorStorageError::Conjugation { .. }));
        let source = std::error::Error::source(&error).unwrap();
        assert_eq!(source.to_string(), message);
        assert!(
            source.source().is_none(),
            "source was wrapped more than once"
        );
        conjugated
    }

    #[test]
    fn authoritative_storage_conjugation_failure_is_deferred_without_detaching() {
        let i = DynIndex::new_dyn(2);
        let native = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
        let inner = EagerTensor::requires_grad_in(native, default_eager_ctx().unwrap());
        let tensor = TensorDynLen::from_inner(vec![i], inner).unwrap();
        let source = match &tensor.storage {
            TensorDynLenStorage::Eager { inner, .. } => Arc::clone(inner),
            TensorDynLenStorage::Compact(payload) => Arc::clone(&payload.payload),
            TensorDynLenStorage::Materialized(_) | TensorDynLenStorage::Deferred { .. } => {
                panic!("tracked eager source expected")
            }
        };

        let conjugated = conjugate_with_injected_failure(
            &tensor,
            Arc::as_ptr(&source),
            "forced authoritative eager conjugation failure",
        );
        assert!(conjugated.tracks_grad());
        assert!(conjugated.is_f64());
        assert!(!conjugated.is_complex());
        assert!(!conjugated.is_diag());
        assert_eq!(conjugated.dims(), vec![2]);

        let error = conjugated.to_storage().unwrap_err();
        let source = std::error::Error::source(&error).unwrap();
        assert_eq!(
            source.to_string(),
            "forced authoritative eager conjugation failure"
        );
        assert!(conjugated.detach().is_err());
    }

    #[test]
    fn structured_payload_conjugation_failure_retains_graph_and_blocks_detached_primal() {
        let i = DynIndex::new_dyn(2);
        let j = DynIndex::new_dyn(2);
        let tensor = TensorDynLen::from_diag(
            vec![i, j],
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
        )
        .unwrap()
        .enable_grad()
        .unwrap();

        let target = Arc::as_ptr(
            &tensor
                .storage
                .compact_payload()
                .expect("tracked compact payload")
                .payload,
        );
        let conjugated = assert_unwrapped_conjugation_error(
            tensor,
            target,
            "forced structured AD conjugation failure",
        );
        assert!(conjugated.tracks_grad());
        assert!(conjugated.detach().is_err());
        assert!(conjugated.clone().enable_grad().is_err());
        assert!(conjugated.sum().is_err());
        assert!(conjugated.grad().is_err());
        assert!(conjugated.clear_grad().is_err());
        assert!(conjugated.maxabs().is_err());
        assert!(conjugated.norm_squared().is_err());

        let twice_conjugated = conjugated.conj();
        let error = twice_conjugated.to_storage().unwrap_err();
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "forced structured AD conjugation failure"
        );
    }

    #[test]
    fn eager_cache_conjugation_failure_is_deferred_with_original_diagnostic() {
        let i = DynIndex::new_dyn(2);
        let j = DynIndex::new_dyn(2);
        let tensor = TensorDynLen::from_diag(vec![i, j], vec![1.0_f64, 2.0]).unwrap();
        tensor.as_inner().unwrap();
        let target = Arc::as_ptr(tensor.eager_cache.get().unwrap());

        let conjugated = assert_unwrapped_conjugation_error(
            tensor,
            target,
            "forced eager cache conjugation failure",
        );
        assert!(!conjugated.tracks_grad());
        assert!(conjugated.detach().is_err());
    }
}
