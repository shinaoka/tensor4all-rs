use anyhow::{anyhow, ensure, Result};
use num_complex::{Complex32, Complex64};
use tenferro::{DType, Tensor as NativeTensor, TensorScalar};

/// Public scalar element types supported by tensor4all dense/diag constructors.
///
/// Implemented for `f32`, `f64`, `Complex32`, and `Complex64`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::TensorElement;
///
/// let t = f64::dense_native_tensor_from_col_major(&[1.0, 2.0], &[2]).unwrap();
/// assert_eq!(t.shape(), &[2]);
///
/// let vals = f64::dense_values_from_native_col_major(&t).unwrap();
/// assert_eq!(vals, vec![1.0, 2.0]);
/// ```
pub trait TensorElement: TensorScalar + Copy + Send + Sync + 'static {
    /// Build a dense native tensor from column-major data.
    /// # Errors
    ///
    /// Returns an error when the data length does not match the dimensions (a
    /// /// shape mismatch) or the backend conversion fails.
    ///
    fn dense_native_tensor_from_col_major(data: &[Self], dims: &[usize]) -> Result<NativeTensor>;

    /// Build a diagonal native tensor from column-major diagonal payload data.
    /// # Errors
    ///
    /// Returns an error when the diagonal payload is incompatible (a shape
    /// /// mismatch) or the backend conversion fails.
    ///
    fn diag_native_tensor_from_col_major(
        data: &[Self],
        logical_rank: usize,
    ) -> Result<NativeTensor>;

    /// Build a rank-0 native tensor.
    /// # Errors
    ///
    /// Returns an error when the scalar cannot be converted (a dtype mismatch or
    /// /// backend failure).
    ///
    fn scalar_native_tensor(value: Self) -> Result<NativeTensor>;

    /// Materialize dense column-major values from a native tensor.
    /// # Errors
    ///
    /// Returns an error when the native tensor cannot be materialized (a dtype
    /// /// mismatch or backend failure).
    ///
    fn dense_values_from_native_col_major(tensor: &NativeTensor) -> Result<Vec<Self>>;

    /// Materialize diagonal values from a dense native tensor.
    /// # Errors
    ///
    /// Returns an error when the native tensor cannot be materialized (a dtype
    /// /// mismatch or backend failure).
    ///
    fn diag_values_from_native_temp(tensor: &NativeTensor) -> Result<Vec<Self>>;
}

fn tensor_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::C32 => "c32",
        DType::C64 => "c64",
    }
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("dimension product overflow"))
    })
}

fn dense_diagonal_values<T: Copy + Default>(diag: &[T], logical_rank: usize) -> Result<Vec<T>> {
    ensure!(
        logical_rank >= 1,
        "diagonal tensor construction requires at least one logical axis"
    );
    let diag_len = diag.len();
    let dims = vec![diag_len; logical_rank];
    let total_len = checked_product(&dims)?;
    let mut dense = vec![T::default(); total_len];
    let diagonal_stride = (0..logical_rank)
        .scan(1usize, |stride, _| {
            let current = *stride;
            *stride = stride.saturating_mul(diag_len);
            Some(current)
        })
        .sum::<usize>();
    for (i, value) in diag.iter().copied().enumerate() {
        dense[i * diagonal_stride] = value;
    }
    Ok(dense)
}

macro_rules! impl_tensor_element {
    ($ty:ty, $dtype:expr) => {
        impl TensorElement for $ty {
            fn dense_native_tensor_from_col_major(
                data: &[Self],
                dims: &[usize],
            ) -> Result<NativeTensor> {
                let expected_len: usize = checked_product(dims)?;
                ensure!(
                    data.len() == expected_len,
                    "dense tensor len {} does not match dims {:?} (expected {})",
                    data.len(),
                    dims,
                    expected_len
                );
                Ok(NativeTensor::from_vec_col_major(
                    dims.to_vec(),
                    data.to_vec(),
                )?)
            }

            fn diag_native_tensor_from_col_major(
                data: &[Self],
                logical_rank: usize,
            ) -> Result<NativeTensor> {
                let dims = vec![data.len(); logical_rank];
                let dense = dense_diagonal_values(data, logical_rank)?;
                Self::dense_native_tensor_from_col_major(&dense, &dims)
            }

            fn scalar_native_tensor(value: Self) -> Result<NativeTensor> {
                Ok(NativeTensor::from_vec_col_major(vec![], vec![value])?)
            }

            fn dense_values_from_native_col_major(tensor: &NativeTensor) -> Result<Vec<Self>> {
                tensor
                    .as_slice::<Self>()
                    .map(|values| values.to_vec())
                    .map_err(|_| {
                        anyhow!(
                            "tensor dtype mismatch: expected {}, got {}",
                            tensor_dtype_name($dtype),
                            tensor_dtype_name(tensor.dtype())
                        )
                    })
            }

            fn diag_values_from_native_temp(tensor: &NativeTensor) -> Result<Vec<Self>> {
                let shape = tensor.shape();
                ensure!(
                    !shape.is_empty(),
                    "diagonal extraction requires rank >= 1, got scalar tensor"
                );
                let diag_len = shape[0];
                ensure!(
                    shape.iter().all(|&dim| dim == diag_len),
                    "expected square/equal dims for diagonal extraction, got {:?}",
                    shape
                );
                let dense = Self::dense_values_from_native_col_major(tensor)?;
                let diagonal_stride = (0..shape.len())
                    .scan(1usize, |stride, _| {
                        let current = *stride;
                        *stride = stride.saturating_mul(diag_len);
                        Some(current)
                    })
                    .sum::<usize>();
                Ok((0..diag_len).map(|i| dense[i * diagonal_stride]).collect())
            }
        }
    };
}

impl_tensor_element!(f32, DType::F32);
impl_tensor_element!(f64, DType::F64);
impl_tensor_element!(Complex32, DType::C32);
impl_tensor_element!(Complex64, DType::C64);
