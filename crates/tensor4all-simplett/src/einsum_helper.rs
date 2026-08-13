use std::error::Error;
use std::fmt;

use crate::traits::TTScalar;
use tenferro_einsum::Subscripts;
use tenferro_tensor::{Tensor, TensorScalar, TypedTensor};
use tensor4all_tensorbackend::einsum_native_tensors;

pub(crate) type Result<T> = std::result::Result<T, EinsumHelperError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EinsumHelperError {
    ShapeElementCountOverflow {
        shape: Vec<usize>,
    },
    DataLengthMismatch {
        shape: Vec<usize>,
        expected: usize,
        got: usize,
    },
    ReshapeElementCountMismatch {
        source_shape: Vec<usize>,
        target_shape: Vec<usize>,
        source_elements: usize,
        target_elements: usize,
    },
    VectorLengthMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    MatrixLengthMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    Backend {
        subscripts: String,
        message: String,
    },
}

impl fmt::Display for EinsumHelperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeElementCountOverflow { shape } => {
                write!(f, "shape element count overflow for shape {shape:?}")
            }
            Self::DataLengthMismatch {
                shape,
                expected,
                got,
            } => write!(
                f,
                "tensor data length mismatch for shape {shape:?}: expected {expected}, got {got}"
            ),
            Self::ReshapeElementCountMismatch {
                source_shape,
                target_shape,
                source_elements,
                target_elements,
            } => write!(
                f,
                "tensor reshape element count mismatch from {source_shape:?} to {target_shape:?}: source has {source_elements}, target has {target_elements}"
            ),
            Self::VectorLengthMismatch { op, expected, got } => {
                write!(f, "{op} length mismatch: expected {expected}, got {got}")
            }
            Self::MatrixLengthMismatch { op, expected, got } => {
                write!(f, "{op} length mismatch: expected {expected}, got {got}")
            }
            Self::Backend {
                subscripts,
                message,
            } => write!(f, "einsum failed for '{subscripts}': {message}"),
        }
    }
}

impl Error for EinsumHelperError {}

fn shape_element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| EinsumHelperError::ShapeElementCountOverflow {
                shape: shape.to_vec(),
            })
    })
}

pub(crate) fn einsum_tensors<T: EinsumScalar>(
    subscripts: &str,
    operands: &[&TypedTensor<T>],
) -> Result<TypedTensor<T>> {
    let parsed = Subscripts::parse(subscripts).map_err(|err| EinsumHelperError::Backend {
        subscripts: subscripts.to_string(),
        message: err.to_string(),
    })?;
    let tensors: Vec<Tensor> = operands
        .iter()
        .map(|tensor| T::into_tensor(tensor.shape().to_vec(), tensor.host_data().to_vec()))
        .collect();
    let input_ids = parsed
        .inputs
        .iter()
        .map(|ids| ids.iter().map(|&id| id as usize).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let operand_refs = tensors
        .iter()
        .zip(input_ids.iter())
        .map(|(tensor, ids)| (tensor, ids.as_slice()))
        .collect::<Vec<_>>();
    let output_ids = parsed
        .output
        .iter()
        .map(|&id| id as usize)
        .collect::<Vec<_>>();

    let result = einsum_native_tensors(&operand_refs, &output_ids).map_err(|err| {
        EinsumHelperError::Backend {
            subscripts: subscripts.to_string(),
            message: err.to_string(),
        }
    })?;
    let actual = result.dtype();
    T::try_into_typed(result).ok_or_else(|| EinsumHelperError::Backend {
        subscripts: subscripts.to_string(),
        message: format!(
            "dtype mismatch: result has {actual:?}, expected {:?}",
            T::dtype()
        ),
    })
}

pub(crate) fn typed_tensor_from_col_major_slice<T: TensorScalar>(
    data: &[T],
    shape: &[usize],
) -> Result<TypedTensor<T>> {
    let expected = shape_element_count(shape)?;
    if expected != data.len() {
        return Err(EinsumHelperError::DataLengthMismatch {
            shape: shape.to_vec(),
            expected,
            got: data.len(),
        });
    }

    Ok(TypedTensor::from_vec_col_major(
        shape.to_vec(),
        data.to_vec(),
    ))
}

pub(crate) fn typed_tensor_reshape<T: TensorScalar>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
) -> Result<TypedTensor<T>> {
    let target_elements = shape_element_count(shape)?;
    let source_elements = shape_element_count(tensor.shape())?;
    if target_elements != source_elements {
        return Err(EinsumHelperError::ReshapeElementCountMismatch {
            source_shape: tensor.shape().to_vec(),
            target_shape: shape.to_vec(),
            source_elements,
            target_elements,
        });
    }

    Ok(TypedTensor::from_vec_col_major(
        shape.to_vec(),
        tensor.host_data().to_vec(),
    ))
}

pub(crate) fn tensor_to_col_major_vec<T: TensorScalar>(tensor: &TypedTensor<T>) -> Vec<T> {
    tensor.host_data().to_vec()
}

pub(crate) fn row_vector_times_matrix<T: TTScalar + EinsumScalar>(
    vector: &[T],
    matrix: &[T],
    rows: usize,
    cols: usize,
) -> Result<Vec<T>> {
    if vector.len() != rows {
        return Err(EinsumHelperError::VectorLengthMismatch {
            op: "row vector",
            expected: rows,
            got: vector.len(),
        });
    }

    let expected_matrix_len =
        rows.checked_mul(cols)
            .ok_or_else(|| EinsumHelperError::ShapeElementCountOverflow {
                shape: vec![rows, cols],
            })?;
    if matrix.len() != expected_matrix_len {
        return Err(EinsumHelperError::MatrixLengthMismatch {
            op: "row vector times matrix",
            expected: expected_matrix_len,
            got: matrix.len(),
        });
    }

    // Direct loop: `einsum_tensors` re-traces and re-compiles the graph on
    // every call (the bridge clears the compile cache for safety), which
    // costs ~70 us even for a 2x2 product. TTCache's environment recursion
    // calls this thousands of times per global-pivot guard search, so the
    // mat-vec is computed inline.
    Ok((0..cols)
        .map(|c| {
            (0..rows)
                .map(|r| vector[r] * matrix[r + c * rows])
                .fold(T::zero(), |acc, x| acc + x)
        })
        .collect())
}

pub(crate) fn matrix_times_col_vector<T: TTScalar + EinsumScalar>(
    matrix: &[T],
    rows: usize,
    cols: usize,
    vector: &[T],
) -> Result<Vec<T>> {
    let expected_matrix_len =
        rows.checked_mul(cols)
            .ok_or_else(|| EinsumHelperError::ShapeElementCountOverflow {
                shape: vec![rows, cols],
            })?;
    if matrix.len() != expected_matrix_len {
        return Err(EinsumHelperError::MatrixLengthMismatch {
            op: "matrix times column vector",
            expected: expected_matrix_len,
            got: matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(EinsumHelperError::VectorLengthMismatch {
            op: "column vector",
            expected: cols,
            got: vector.len(),
        });
    }

    // Direct loop, same rationale as `row_vector_times_matrix`: the einsum
    // dispatch recompiles the graph on every call, which dominates small
    // environment contractions.
    Ok((0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| matrix[r + c * rows] * vector[c])
                .fold(T::zero(), |acc, x| acc + x)
        })
        .collect())
}

/// Scalar types supported by the tenferro-backed einsum helpers used in
/// `tensor4all-simplett`.
pub trait EinsumScalar: TensorScalar + Sized {}

macro_rules! impl_einsum_scalar {
    ($($t:ty),*) => {
        $(impl EinsumScalar for $t {})*
    };
}

impl_einsum_scalar!(f64, f32, num_complex::Complex64, num_complex::Complex32);
