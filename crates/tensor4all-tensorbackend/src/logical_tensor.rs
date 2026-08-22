//! Backend-free logical tensor snapshots for transfer between execution domains.

use num_complex::{Complex32, Complex64};
use tenferro::{DType, Tensor, TensorRead};
use tenferro_tensor::BackendSessionHost;

use crate::CpuExecutionContext;

/// Owned column-major scalar payload for a [`LogicalTensor`].
///
/// The enum contains values only; it cannot carry an executor, backend,
/// runtime, cache, pointer, or address identity.
#[derive(Clone, Debug, PartialEq)]
pub enum LogicalTensorData {
    /// 32-bit real values.
    F32(Vec<f32>),
    /// 64-bit real values.
    F64(Vec<f64>),
    /// 32-bit signed integer values.
    I32(Vec<i32>),
    /// 64-bit signed integer values.
    I64(Vec<i64>),
    /// Boolean values.
    Bool(Vec<bool>),
    /// 32-bit complex values.
    C32(Vec<Complex32>),
    /// 64-bit complex values.
    C64(Vec<Complex64>),
}

impl LogicalTensorData {
    /// Return the scalar dtype represented by this payload.
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    /// Return the number of logical scalar values.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::C32(values) => values.len(),
            Self::C64(values) => values.len(),
        }
    }

    /// Return whether the payload contains no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Error returned while validating, snapshotting, or reconstructing a logical tensor.
#[derive(Debug, thiserror::Error)]
pub enum LogicalTensorError {
    /// The shape product overflowed `usize`.
    #[error("logical tensor shape product overflows usize: {shape:?}")]
    ShapeOverflow {
        /// Rejected tensor shape.
        shape: Vec<usize>,
    },
    /// The payload length does not match the shape.
    #[error(
        "logical tensor element count mismatch: shape {shape:?} requires {expected}, got {actual}"
    )]
    ElementCountMismatch {
        /// Tensor shape.
        shape: Vec<usize>,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Tenferro could not read or construct the host tensor.
    #[error("logical tensor {operation} failed: {source}")]
    Tensor {
        /// Operation that failed.
        operation: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: tenferro_tensor::Error,
    },
}

/// Backend-free, dtype-preserving logical host tensor in column-major order.
///
/// Use [`CpuExecutionContext::reconstruct`] in the receiving execution domain.
/// The adapter, not tensor4all, owns serialization and transport.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{LogicalTensor, LogicalTensorData};
///
/// let tensor = LogicalTensor::new(
///     vec![2, 2],
///     LogicalTensorData::F64(vec![1.0, 2.0, 3.0, 4.0]),
/// )?;
/// assert_eq!(tensor.shape(), &[2, 2]);
/// assert_eq!(tensor.data().len(), 4);
/// # Ok::<(), tensor4all_tensorbackend::LogicalTensorError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalTensor {
    shape: Vec<usize>,
    data: LogicalTensorData,
}

impl LogicalTensor {
    /// Create a validated column-major logical tensor snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`LogicalTensorError::ShapeOverflow`] when the shape product
    /// overflows, or [`LogicalTensorError::ElementCountMismatch`] when the data
    /// length differs from that product.
    pub fn new(shape: Vec<usize>, data: LogicalTensorData) -> Result<Self, LogicalTensorError> {
        let expected = shape
            .iter()
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim).ok_or(()));
        let expected = expected.map_err(|()| LogicalTensorError::ShapeOverflow {
            shape: shape.clone(),
        })?;
        if data.len() != expected {
            return Err(LogicalTensorError::ElementCountMismatch {
                shape,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// Snapshot a native host tensor without retaining execution identity.
    ///
    /// # Errors
    ///
    /// Returns [`LogicalTensorError::Tensor`] if the native tensor is not
    /// host-readable with its declared dtype.
    pub fn from_native(tensor: &Tensor) -> Result<Self, LogicalTensorError> {
        let data = match tensor.dtype() {
            DType::F32 => LogicalTensorData::F32(
                tensor
                    .as_slice::<f32>()
                    .map_err(|source| LogicalTensorError::Tensor {
                        operation: "snapshot",
                        source,
                    })?
                    .to_vec(),
            ),
            DType::F64 => LogicalTensorData::F64(
                tensor
                    .as_slice::<f64>()
                    .map_err(|source| LogicalTensorError::Tensor {
                        operation: "snapshot",
                        source,
                    })?
                    .to_vec(),
            ),
            DType::I32 => LogicalTensorData::I32(
                tensor
                    .as_slice::<i32>()
                    .map_err(|source| LogicalTensorError::Tensor {
                        operation: "snapshot",
                        source,
                    })?
                    .to_vec(),
            ),
            DType::I64 => LogicalTensorData::I64(
                tensor
                    .as_slice::<i64>()
                    .map_err(|source| LogicalTensorError::Tensor {
                        operation: "snapshot",
                        source,
                    })?
                    .to_vec(),
            ),
            DType::Bool => LogicalTensorData::Bool(
                tensor
                    .as_slice::<bool>()
                    .map_err(|source| LogicalTensorError::Tensor {
                        operation: "snapshot",
                        source,
                    })?
                    .to_vec(),
            ),
            DType::C32 => LogicalTensorData::C32(
                tensor
                    .as_slice::<Complex32>()
                    .map_err(|source| LogicalTensorError::Tensor {
                        operation: "snapshot",
                        source,
                    })?
                    .to_vec(),
            ),
            DType::C64 => LogicalTensorData::C64(
                tensor
                    .as_slice::<Complex64>()
                    .map_err(|source| LogicalTensorError::Tensor {
                        operation: "snapshot",
                        source,
                    })?
                    .to_vec(),
            ),
        };
        Self::new(tensor.shape().to_vec(), data)
    }

    /// Return the logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the scalar dtype.
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Return the owned scalar payload.
    pub fn data(&self) -> &LogicalTensorData {
        &self.data
    }

    fn to_native(&self) -> Result<Tensor, LogicalTensorError> {
        let result = match &self.data {
            LogicalTensorData::F32(values) => {
                Tensor::from_vec_col_major(self.shape.clone(), values.clone())
            }
            LogicalTensorData::F64(values) => {
                Tensor::from_vec_col_major(self.shape.clone(), values.clone())
            }
            LogicalTensorData::I32(values) => {
                Tensor::from_vec_col_major(self.shape.clone(), values.clone())
            }
            LogicalTensorData::I64(values) => {
                Tensor::from_vec_col_major(self.shape.clone(), values.clone())
            }
            LogicalTensorData::Bool(values) => {
                Tensor::from_vec_col_major(self.shape.clone(), values.clone())
            }
            LogicalTensorData::C32(values) => {
                Tensor::from_vec_col_major(self.shape.clone(), values.clone())
            }
            LogicalTensorData::C64(values) => {
                Tensor::from_vec_col_major(self.shape.clone(), values.clone())
            }
        };
        result.map_err(|source| LogicalTensorError::Tensor {
            operation: "reconstruction",
            source,
        })
    }
}

impl CpuExecutionContext {
    /// Reconstruct a logical host tensor in this receiving execution domain.
    ///
    /// The returned value carries only host tensor data. Its first plain, graph,
    /// or eager operation must use this same explicit context.
    ///
    /// # Errors
    ///
    /// Returns [`LogicalTensorError`] if tenferro rejects the validated shape or
    /// dtype payload.
    pub fn reconstruct(&self, tensor: &LogicalTensor) -> Result<Tensor, LogicalTensorError> {
        let host = tensor.to_native()?;
        self.with_backend(|backend| {
            backend.with_backend_session(|session| {
                session.upload_host_tensor(TensorRead::from_tensor(&host))
            })
        })
        .map_err(|source| LogicalTensorError::Tensor {
            operation: "target-context upload",
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_cpu::CpuBackend;

    #[test]
    fn round_trip_preserves_shape_dtype_and_values_in_target_context() {
        let native = Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(1.0, 2.0),
                Complex32::new(3.0, 4.0),
                Complex32::new(5.0, 6.0),
                Complex32::new(7.0, 8.0),
            ],
        )
        .unwrap();
        let logical = LogicalTensor::from_native(&native).unwrap();
        let target = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
        let rebuilt = target.reconstruct(&logical).unwrap();

        assert_eq!(rebuilt.shape(), &[2, 2]);
        assert_eq!(rebuilt.dtype(), DType::C32);
        assert_eq!(
            rebuilt.as_slice::<Complex32>().unwrap(),
            native.as_slice::<Complex32>().unwrap()
        );
    }

    #[test]
    fn validation_rejects_overflow_and_length_mismatch() {
        assert!(matches!(
            LogicalTensor::new(vec![usize::MAX, 2], LogicalTensorData::F64(Vec::new())),
            Err(LogicalTensorError::ShapeOverflow { .. })
        ));
        assert!(matches!(
            LogicalTensor::new(vec![2], LogicalTensorData::F64(vec![1.0])),
            Err(LogicalTensorError::ElementCountMismatch { .. })
        ));
    }
}
