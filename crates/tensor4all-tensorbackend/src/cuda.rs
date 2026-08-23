//! Explicit CUDA execution context and transfer boundaries.

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro::Tensor;
use tenferro_ad::{ContextId, EagerRuntime};
use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
use tenferro_tensor::{
    AllocationDomainId, BackendSessionHost, DeviceId, DeviceKind, GpuBackendKind, MemoryKind,
    Placement, TensorRead,
};

/// The only CUDA ordinal selected by [`CudaExecutionContext`].
pub const CUDA_ORDINAL: u32 = 0;

/// Error returned by explicit CUDA context creation, placement validation, and
/// device transfer.
///
/// The original provider, runtime, or tensor diagnostic is retained as the
/// standard error source where one exists. CUDA operations never fall back to
/// a CPU backend or perform an implicit transfer.
///
/// # Remedies
///
/// - Initialization failures: make visible CUDA ordinal 0 and its runtime
///   libraries available, then retry [`CudaExecutionContext::new`].
/// - Placement failures: use [`CudaExecutionContext::upload_cuda`] for a host
///   tensor, or use one context consistently for resident values.
/// - Transfer failures: keep the explicit upload/download boundary and inspect
///   the retained source error.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::CudaExecutionContextError;
///
/// let error = CudaExecutionContextError::Initialization {
///     component: "CUDA backend",
///     source: Arc::new(std::io::Error::other("driver unavailable")),
/// };
/// assert!(error.source().is_some());
/// assert!(error.to_string().contains("driver unavailable"));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaExecutionContextError {
    /// CUDA device or eager-runtime initialization failed.
    #[error(
        "failed to initialize CUDA {component}: {source}; make visible CUDA ordinal 0 and retry"
    )]
    Initialization {
        /// Context component that failed to initialize.
        component: &'static str,
        /// Original provider or runtime diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// An explicit upload or download failed.
    #[error("CUDA {operation} transfer failed: {source}; keep the transfer explicit and inspect the source error")]
    Transfer {
        /// Transfer operation that failed.
        operation: &'static str,
        /// Original backend transfer diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// A host/device placement does not satisfy the context contract.
    #[error("CUDA {operation} placement mismatch: expected a dense value on visible CUDA ordinal 0; use explicit upload_cuda/download or one context consistently")]
    PlacementMismatch {
        /// Operation that observed the mismatch.
        operation: &'static str,
    },
    /// A resident allocation belongs to another CUDA allocation domain.
    #[error("CUDA {operation} received a foreign allocation domain; use one CudaExecutionContext for upload, computation, and download")]
    ForeignAllocation {
        /// Operation that observed the foreign allocation.
        operation: &'static str,
    },
    /// An input violates the host-only upload boundary.
    #[error("CUDA {operation} received an unsupported input: {reason}; {remedy}")]
    UnsupportedInput {
        /// Operation that rejected the input.
        operation: &'static str,
        /// Stable reason for rejecting the input.
        reason: &'static str,
        /// Corrective action for the caller.
        remedy: &'static str,
    },
    /// Synchronization of the context-owned CUDA stream failed.
    #[error("CUDA synchronization failed: {source}; retry after checking the CUDA runtime")]
    Synchronization {
        /// Original CUDA synchronization diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

/// Caller-owned CUDA execution domain for explicit ordinal-0 transfers.
///
/// The context owns one [`CudaBackend`] and lazily creates one shared
/// [`EagerRuntime`] from a clone of that backend. Cloning preserves the
/// provider's runtime and allocation-domain identity. No CPU backend, global
/// CUDA context, or implicit transfer is used.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
/// use std::sync::Arc;
/// use tenferro_ad::EagerRuntime;
/// use tenferro::Tensor;
///
/// let new_context: fn() -> Result<CudaExecutionContext, CudaExecutionContextError> =
///     CudaExecutionContext::new;
/// let upload: fn(&CudaExecutionContext, &Tensor) -> Result<Tensor, CudaExecutionContextError> =
///     CudaExecutionContext::upload_cuda;
/// let eager: fn(&CudaExecutionContext) -> Result<Arc<EagerRuntime>, CudaExecutionContextError> =
///     CudaExecutionContext::eager_runtime;
/// assert_eq!(std::mem::size_of_val(&new_context), std::mem::size_of::<fn() -> Result<CudaExecutionContext, CudaExecutionContextError>>());
/// assert_eq!(std::mem::size_of_val(&upload), std::mem::size_of::<fn(&CudaExecutionContext, &Tensor) -> Result<Tensor, CudaExecutionContextError>>());
/// assert_eq!(std::mem::size_of_val(&eager), std::mem::size_of::<fn(&CudaExecutionContext) -> Result<Arc<EagerRuntime>, CudaExecutionContextError>>());
/// ```
pub struct CudaExecutionContext {
    backend: Mutex<CudaBackend>,
    eager: OnceLock<Result<Arc<EagerRuntime>, CudaExecutionContextError>>,
    device_name: String,
}

impl fmt::Debug for CudaExecutionContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaExecutionContext")
            .field("ordinal", &CUDA_ORDINAL)
            .field("device_name", &self.device_name)
            .field("eager_initialized", &self.eager.get().is_some())
            .finish_non_exhaustive()
    }
}

impl CudaExecutionContext {
    /// Create a context for visible CUDA ordinal 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError, CUDA_ORDINAL};
    /// let constructor: fn() -> Result<CudaExecutionContext, CudaExecutionContextError> = CudaExecutionContext::new;
    /// assert_eq!(CUDA_ORDINAL, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Initialization`] when device
    /// discovery or CUDA runtime initialization fails.
    pub fn new() -> Result<Self, CudaExecutionContextError> {
        let device = cuda_devices()
            .map_err(|source| CudaExecutionContextError::Initialization {
                component: "device discovery",
                source: Arc::new(source),
            })?
            .into_iter()
            .find(|device| device.id().ordinal() == CUDA_ORDINAL)
            .ok_or_else(|| CudaExecutionContextError::Initialization {
                component: "device discovery",
                source: Arc::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "visible CUDA ordinal 0 is unavailable",
                )),
            })?;
        let device_name = device.name().to_owned();
        let backend = CudaBackend::new(device.id()).map_err(|source| {
            CudaExecutionContextError::Initialization {
                component: "backend",
                source: Arc::new(source),
            }
        })?;
        Ok(Self {
            backend: Mutex::new(backend),
            eager: OnceLock::new(),
            device_name,
        })
    }

    /// Return the only supported visible CUDA ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CUDA_ORDINAL};
    /// let accessor: fn(&CudaExecutionContext) -> u32 = CudaExecutionContext::ordinal;
    /// assert_eq!(CUDA_ORDINAL, 0);
    /// ```
    pub const fn ordinal(&self) -> u32 {
        CUDA_ORDINAL
    }

    /// Return the discovered name of visible CUDA ordinal 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CudaExecutionContext;
    /// let accessor: for<'a> fn(&'a CudaExecutionContext) -> &'a str = CudaExecutionContext::device_name;
    /// assert!(std::mem::size_of_val(&accessor) > 0);
    /// ```
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Return this context's eager runtime identity, initializing it lazily.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let accessor: fn(&CudaExecutionContext) -> Result<tenferro_ad::ContextId, CudaExecutionContextError> = CudaExecutionContext::context_id;
    /// assert!(std::mem::size_of_val(&accessor) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Initialization`] when the CUDA
    /// eager runtime or its runtime registration cannot be created.
    pub fn context_id(&self) -> Result<ContextId, CudaExecutionContextError> {
        self.eager_runtime().map(|runtime| runtime.id())
    }

    /// Return the context-owned CUDA eager runtime.
    ///
    /// Repeated calls return the same [`Arc`] and preserve the runtime and
    /// allocation identity shared with the context-owned backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let accessor: fn(&CudaExecutionContext) -> Result<Arc<tenferro_ad::EagerRuntime>, CudaExecutionContextError> = CudaExecutionContext::eager_runtime;
    /// assert!(std::mem::size_of_val(&accessor) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Initialization`] when eager
    /// runtime registration fails.
    pub fn eager_runtime(&self) -> Result<Arc<EagerRuntime>, CudaExecutionContextError> {
        self.eager
            .get_or_init(|| {
                let backend = self.backend_clone();
                EagerRuntime::with_cuda_backend(backend).map_err(|source| {
                    CudaExecutionContextError::Initialization {
                        component: "eager runtime",
                        source: Arc::new(source),
                    }
                })
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    /// Upload a host tensor explicitly into this CUDA context.
    ///
    /// The source must be host-backed. A resident CUDA or foreign-backend
    /// tensor is rejected rather than copied implicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let upload: fn(&CudaExecutionContext, &tenferro::Tensor) -> Result<tenferro::Tensor, CudaExecutionContextError> = CudaExecutionContext::upload_cuda;
    /// assert!(std::mem::size_of_val(&upload) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::UnsupportedInput`] for a
    /// non-host source, [`CudaExecutionContextError::Transfer`] for a backend
    /// transfer failure, or [`CudaExecutionContextError::PlacementMismatch`]
    /// for invalid placement metadata.
    pub fn upload_cuda(&self, tensor: &Tensor) -> Result<Tensor, CudaExecutionContextError> {
        let read = TensorRead::from_tensor(tensor);
        self.validate_host_placement_metadata(read.placement(), read.allocation_domain())?;
        self.with_backend_session(|session| session.upload_host_tensor(read))
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "upload",
                source: Arc::new(source),
            })
    }

    /// Download a tensor resident on this exact CUDA context into host storage.
    ///
    /// This method never accepts host tensors and never downloads from a
    /// different allocation domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let download: fn(&CudaExecutionContext, &tenferro::Tensor) -> Result<tenferro::Tensor, CudaExecutionContextError> = CudaExecutionContext::download;
    /// assert!(std::mem::size_of_val(&download) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] for host,
    /// non-CUDA, or nonzero-ordinal inputs,
    /// [`CudaExecutionContextError::ForeignAllocation`] for a tensor owned by
    /// another CUDA context, or [`CudaExecutionContextError::Transfer`] when
    /// the explicit download fails.
    pub fn download(&self, tensor: &Tensor) -> Result<Tensor, CudaExecutionContextError> {
        self.validate_cuda_placement(tensor)?;
        let read = TensorRead::from_tensor(tensor);
        self.with_backend_session(|session| session.download_to_host(read))
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "download",
                source: Arc::new(source),
            })
    }

    /// Validate an owned tensor's placement without transferring or reading it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let validate: fn(&CudaExecutionContext, &tenferro::Tensor) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::validate_cuda_placement;
    /// assert!(std::mem::size_of_val(&validate) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] when the tensor
    /// is not CUDA-resident on visible ordinal 0, or
    /// [`CudaExecutionContextError::ForeignAllocation`] when it belongs to a
    /// different CUDA context.
    pub fn validate_cuda_placement(
        &self,
        tensor: &Tensor,
    ) -> Result<(), CudaExecutionContextError> {
        let read = TensorRead::from_tensor(tensor);
        if read.backend_family() != Some("cuda") {
            return Err(CudaExecutionContextError::PlacementMismatch {
                operation: "validate CUDA placement",
            });
        }
        self.validate_cuda_placement_metadata(read.placement(), read.allocation_domain())
    }

    /// Validate placement metadata obtained from a borrowed eager value.
    ///
    /// This is a metadata-only check: it performs no host access and no device
    /// transfer. It exists so higher-level wrappers can validate an
    /// [`EagerRuntime`] value without first materializing an owned [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// use tenferro_tensor::{AllocationDomainId, Placement};
    /// let validate: fn(&CudaExecutionContext, &Placement, Option<AllocationDomainId>) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::validate_cuda_placement_metadata;
    /// assert!(std::mem::size_of_val(&validate) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] for a
    /// non-CUDA-ordinal-0 placement and
    /// [`CudaExecutionContextError::ForeignAllocation`] for a different
    /// allocation domain.
    pub fn validate_cuda_placement_metadata(
        &self,
        placement: &Placement,
        allocation_domain: Option<AllocationDomainId>,
    ) -> Result<(), CudaExecutionContextError> {
        if !is_cuda_ordinal_zero_placement(placement) {
            return Err(CudaExecutionContextError::PlacementMismatch {
                operation: "validate CUDA placement",
            });
        }
        let expected = self.with_backend(|backend| backend.runtime().allocation_domain());
        if allocation_domain != Some(expected) {
            return Err(CudaExecutionContextError::ForeignAllocation {
                operation: "validate CUDA placement",
            });
        }
        Ok(())
    }

    /// Validate metadata for a host-only upload source.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// use tenferro_tensor::{AllocationDomainId, Placement};
    /// let validate: fn(&CudaExecutionContext, &Placement, Option<AllocationDomainId>) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::validate_host_placement_metadata;
    /// assert!(std::mem::size_of_val(&validate) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::UnsupportedInput`] when the source
    /// carries device or backend allocation metadata.
    pub fn validate_host_placement_metadata(
        &self,
        placement: &Placement,
        allocation_domain: Option<AllocationDomainId>,
    ) -> Result<(), CudaExecutionContextError> {
        validate_host_placement_metadata(placement, allocation_domain)
    }

    /// Synchronize work submitted through this context's CUDA runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let synchronize: fn(&CudaExecutionContext) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::synchronize;
    /// assert!(std::mem::size_of_val(&synchronize) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Synchronization`] when the CUDA
    /// stream cannot be synchronized.
    pub fn synchronize(&self) -> Result<(), CudaExecutionContextError> {
        self.with_backend(|backend| backend.runtime().synchronize())
            .map_err(|source| CudaExecutionContextError::Synchronization {
                source: Arc::new(source),
            })
    }

    fn with_backend<R>(&self, f: impl FnOnce(&mut CudaBackend) -> R) -> R {
        let mut backend = match self.backend.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut backend)
    }

    fn backend_clone(&self) -> CudaBackend {
        self.with_backend(|backend| backend.clone())
    }

    fn with_backend_session<R: Send>(
        &self,
        f: impl FnOnce(&mut dyn tenferro_tensor::BackendSession) -> R + Send,
    ) -> R {
        self.with_backend(|backend| backend.with_backend_session(f))
    }
}

fn validate_host_placement_metadata(
    placement: &Placement,
    allocation_domain: Option<AllocationDomainId>,
) -> Result<(), CudaExecutionContextError> {
    if placement.device.is_some()
        || matches!(placement.memory_kind, MemoryKind::Device)
        || allocation_domain.is_some()
    {
        return Err(CudaExecutionContextError::UnsupportedInput {
            operation: "upload",
            reason: "the source is not host-resident",
            remedy: "download it explicitly or provide a host tensor",
        });
    }
    Ok(())
}

fn is_cuda_ordinal_zero_placement(placement: &Placement) -> bool {
    placement.memory_kind == MemoryKind::Device
        && placement.device.as_ref()
            == Some(&DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: CUDA_ORDINAL as usize,
            })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn source_chain_is_preserved_for_initialization() {
        let error = CudaExecutionContextError::Initialization {
            component: "backend",
            source: Arc::new(std::io::Error::other("driver unavailable")),
        };
        assert!(error.source().is_some());
        assert_eq!(error.source().unwrap().to_string(), "driver unavailable");
    }

    #[test]
    fn metadata_rejections_do_not_need_a_cuda_context() {
        let host = Placement::default();
        assert!(validate_host_placement_metadata(&host, None).is_ok());
        assert!(!is_cuda_ordinal_zero_placement(&host));

        let cuda = Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: CUDA_ORDINAL as usize,
            }),
            cpu_affinity: None,
        };
        assert!(is_cuda_ordinal_zero_placement(&cuda));
        assert!(matches!(
            validate_host_placement_metadata(&cuda, None),
            Err(CudaExecutionContextError::UnsupportedInput { .. })
        ));
    }
}
