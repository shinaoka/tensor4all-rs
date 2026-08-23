//! CUDA transfer boundaries for [`IdxTensor`].

use std::sync::Arc;

use tenferro_ad::EagerTensor;
use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError, StorageKind};

use super::idx_tensor::{IdxTensor, TensorStorageError};

/// Error returned by explicit [`IdxTensor`] CUDA transfer and residency checks.
///
/// CUDA transfer is deliberately limited to untracked, logically dense values.
/// The error retains deferred-storage and context failures as typed sources; it
/// never falls back to a CPU transfer or exposes runtime identifiers.
///
/// # Examples
///
/// ```
/// use tensor4all_core::defaults::IdxTensorCudaError;
///
/// let error = IdxTensorCudaError::Tracked { operation: "upload" };
/// assert!(error.to_string().contains("tracked"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum IdxTensorCudaError {
    /// The tensor has a deferred storage diagnostic that must be reported first.
    #[error("IdxTensor CUDA {operation} rejected deferred storage: {source}")]
    DeferredStorage {
        /// Operation that observed the deferred storage.
        operation: &'static str,
        /// Original storage diagnostic.
        #[source]
        source: TensorStorageError,
    },
    /// CUDA transfer does not preserve automatic-differentiation tracking.
    #[error(
        "IdxTensor CUDA {operation} rejects tracked values; provide an untracked dense tensor"
    )]
    Tracked {
        /// Operation that rejected the tracked tensor.
        operation: &'static str,
    },
    /// CUDA transfer does not accept diagonal or general structured storage.
    #[error("IdxTensor CUDA {operation} requires dense storage, got {kind:?}")]
    UnsupportedStorage {
        /// Operation that rejected the storage layout.
        operation: &'static str,
        /// Logical storage layout observed before any materialization.
        kind: StorageKind,
    },
    /// CUDA transfer requires canonical one-class-per-axis metadata.
    #[error("IdxTensor CUDA {operation} requires dense axis classes")]
    NonDenseAxisClasses {
        /// Operation that rejected the axis metadata.
        operation: &'static str,
    },
    /// A residency check found no eager value to inspect.
    #[error("IdxTensor CUDA {operation} requires an eager value")]
    MissingEagerInner {
        /// Operation that required the eager value.
        operation: &'static str,
    },
    /// The supplied CUDA context rejected placement or initialization metadata.
    #[error("IdxTensor CUDA {operation} context operation failed: {source}")]
    Context {
        /// Operation that observed the context failure.
        operation: &'static str,
        /// Original typed context diagnostic.
        #[source]
        source: CudaExecutionContextError,
    },
    /// The tensor belongs to another eager CUDA context.
    #[error("IdxTensor CUDA {operation} received a value from another context")]
    ForeignContext {
        /// Operation that observed the context mismatch.
        operation: &'static str,
    },
    /// Native eager wrapping or explicit transfer preparation failed.
    #[error("IdxTensor CUDA {operation} failed: {source}")]
    Operation {
        /// Operation that failed.
        operation: &'static str,
        /// Original native or eager diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

fn precheck(tensor: &IdxTensor, operation: &'static str) -> Result<(), IdxTensorCudaError> {
    if let Some(source) = tensor.deferred_storage_error() {
        return Err(IdxTensorCudaError::DeferredStorage {
            operation,
            source: source.clone(),
        });
    }
    if tensor.tracks_grad() {
        return Err(IdxTensorCudaError::Tracked { operation });
    }
    let kind = tensor.storage_kind();
    if kind != StorageKind::Dense {
        return Err(IdxTensorCudaError::UnsupportedStorage { operation, kind });
    }
    if !tensor
        .axis_classes()
        .iter()
        .copied()
        .eq(0..tensor.indices().len())
    {
        return Err(IdxTensorCudaError::NonDenseAxisClasses { operation });
    }
    Ok(())
}

fn operation_error<E>(operation: &'static str, source: E) -> IdxTensorCudaError
where
    E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
{
    IdxTensorCudaError::Operation {
        operation,
        source: Arc::from(source.into()),
    }
}

fn context_error(operation: &'static str, source: CudaExecutionContextError) -> IdxTensorCudaError {
    IdxTensorCudaError::Context { operation, source }
}

impl IdxTensor {
    /// Upload this untracked, logically dense tensor into a CUDA context.
    ///
    /// The host value is duplicated before the explicit device transfer. The
    /// returned tensor keeps the original ordered indices and is owned by the
    /// supplied context's eager runtime.
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError`] when storage is deferred, tracked,
    /// structured, not eagerly materializable, unsupported by CUDA, or when the
    /// explicit context transfer fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor, IdxTensorCudaError};
    ///
    /// let upload: fn(
    ///     &IdxTensor,
    ///     &CudaExecutionContext,
    /// ) -> Result<IdxTensor, IdxTensorCudaError> = IdxTensor::upload_cuda;
    /// assert_eq!(
    ///     std::mem::size_of_val(&upload),
    ///     std::mem::size_of::<fn(
    ///         &IdxTensor,
    ///         &CudaExecutionContext,
    ///     ) -> Result<IdxTensor, IdxTensorCudaError>>(),
    /// );
    /// ```
    pub fn upload_cuda(&self, context: &CudaExecutionContext) -> Result<Self, IdxTensorCudaError> {
        precheck(self, "upload")?;
        let native = self
            .cuda_duplicate_native()
            .map_err(|source| operation_error("upload materialization", source))?;
        let uploaded = context
            .upload_cuda(&native)
            .map_err(|source| context_error("upload", source))?;
        let inner = EagerTensor::from_tensor_in(
            uploaded,
            context
                .eager_runtime()
                .map_err(|source| context_error("upload", source))?,
        )
        .map_err(|source| operation_error("upload eager wrapping", source))?;
        Self::from_inner(self.indices.clone(), inner)
            .map_err(|source| operation_error("upload tensor construction", source))
    }

    /// Download this CUDA-resident tensor into the host eager context.
    ///
    /// The residency and context checks are metadata-only. The resident value
    /// is then duplicated in its device allocation domain, downloaded
    /// explicitly, and reconstructed with the original ordered indices.
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError`] when the value is not an untracked dense
    /// tensor resident in the supplied context, or when explicit download or
    /// host reconstruction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor, IdxTensorCudaError};
    ///
    /// let download: fn(
    ///     &IdxTensor,
    ///     &CudaExecutionContext,
    /// ) -> Result<IdxTensor, IdxTensorCudaError> = IdxTensor::download;
    /// assert_eq!(
    ///     std::mem::size_of_val(&download),
    ///     std::mem::size_of::<fn(
    ///         &IdxTensor,
    ///         &CudaExecutionContext,
    ///     ) -> Result<IdxTensor, IdxTensorCudaError>>(),
    /// );
    /// ```
    pub fn download(&self, context: &CudaExecutionContext) -> Result<Self, IdxTensorCudaError> {
        self.validate_cuda_residency(context)?;
        let resident = self
            .cuda_duplicate_native()
            .map_err(|source| operation_error("download materialization", source))?;
        let host = context
            .download(&resident)
            .map_err(|source| context_error("download", source))?;
        Self::from_native(self.indices.clone(), host)
            .map_err(|source| operation_error("download tensor construction", source))
    }

    /// Validate CUDA placement and eager-runtime identity without reading data.
    ///
    /// This checks the existing eager value's context identity and
    /// [`tenferro::TensorRead`] placement metadata. It does not duplicate,
    /// download, or otherwise access host tensor values.
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError`] when the tensor is deferred, tracked,
    /// structured, missing an eager value, placed on a different CUDA context,
    /// or rejected by the supplied context's placement validator.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor, IdxTensorCudaError};
    ///
    /// let validate: fn(
    ///     &IdxTensor,
    ///     &CudaExecutionContext,
    /// ) -> Result<(), IdxTensorCudaError> = IdxTensor::validate_cuda_residency;
    /// assert_eq!(
    ///     std::mem::size_of_val(&validate),
    ///     std::mem::size_of::<fn(
    ///         &IdxTensor,
    ///         &CudaExecutionContext,
    ///     ) -> Result<(), IdxTensorCudaError>>(),
    /// );
    /// ```
    pub fn validate_cuda_residency(
        &self,
        context: &CudaExecutionContext,
    ) -> Result<(), IdxTensorCudaError> {
        precheck(self, "validate CUDA residency")?;
        let inner = self
            .cuda_eager_inner()
            .ok_or(IdxTensorCudaError::MissingEagerInner {
                operation: "validate CUDA residency",
            })?;
        let expected_context = context
            .context_id()
            .map_err(|source| context_error("validate CUDA residency", source))?;
        if inner.ctx_id() != expected_context {
            return Err(IdxTensorCudaError::ForeignContext {
                operation: "validate CUDA residency",
            });
        }
        let read = inner.tensor_read();
        context
            .validate_cuda_placement_metadata(read.placement(), read.allocation_domain())
            .map_err(|source| context_error("validate CUDA residency", source))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tensor4all_tensorbackend::{Storage, StorageKind};

    use super::*;
    use crate::{DynIndex, IdxTensor};

    #[test]
    fn structured_transfer_rejection_needs_no_cuda_context() {
        let i = DynIndex::new_dyn(2);
        let j = DynIndex::new_dyn(2);
        let storage = Arc::new(Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap());
        let tensor = IdxTensor::from_structured_storage(vec![i, j], storage).unwrap();

        assert!(matches!(
            precheck(&tensor, "upload"),
            Err(IdxTensorCudaError::UnsupportedStorage {
                kind: StorageKind::Diagonal,
                ..
            })
        ));
    }

    #[test]
    fn tracked_transfer_rejection_needs_no_cuda_context() {
        let tensor = IdxTensor::scalar(1.0_f64).unwrap().enable_grad().unwrap();

        assert!(matches!(
            precheck(&tensor, "upload"),
            Err(IdxTensorCudaError::Tracked { .. })
        ));
    }
}
