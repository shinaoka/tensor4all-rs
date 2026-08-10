//! Common types and helper functions for quantics transformations.

use crate::error::QuanticsTransformError;
use std::collections::HashMap;
use std::mem::size_of;

use anyhow::Result;
use num_complex::Complex64;
use num_traits::One;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::TensorDynLen;
use tensor4all_simplett::{
    tensor3_from_data, types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain,
};
use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};

/// Type alias for the default index type.
pub type DynIndex = Index<DynId, TagSet>;

/// Boundary condition for quantics transformations.
/// Controls how operators handle values that exceed the representable range
/// `[0, 2^R)`.
/// # Variants
/// - **`Periodic`** (default): Results wrap around modulo 2^R.
///
///   Use when functions are periodic or when wraparound is acceptable.
/// - **`AntiPeriodic`**: Results wrap around modulo 2^R and receive a sign
///
///   `(-1)^q`, where `q` is the integer wrap quotient.
/// - **`Open`**: Out-of-range results produce zeros.
///
///   Use when the function has compact support or when boundary effects matter.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::BoundaryCondition;
/// // Default is Periodic
/// let bc = BoundaryCondition::default();
/// assert_eq!(bc, BoundaryCondition::Periodic);
/// // Periodic: shift(7, 2) in 3-bit (mod 8) wraps to 1
/// // AntiPeriodic: the same wrap receives a -1 sign
/// // Open: shift(7, 2) in 3-bit goes to 9 >= 8, produces zero
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Periodic boundary: operations wrap around mod 2^R.
    ///
    /// Use for periodic functions or when wraparound is desired.
    #[default]
    Periodic,
    /// Anti-periodic boundary: operations wrap around mod 2^R and receive a
    /// sign for each wrap.
    ///
    /// Use for anti-periodic functions where `f(x + q * 2^R) = (-1)^q f(x)`.
    AntiPeriodic,
    /// Open boundary: operations beyond `[0, 2^R)` return zero.
    ///
    /// Use when the function has compact support or boundary effects matter.
    Open,
}

/// Direction for carry propagation in binary arithmetic operations.
/// This is an internal detail of how binary arithmetic (addition, subtraction)
/// is implemented in the MPO construction. Most users do not need to set this
/// directly.
/// # Variants
/// - **`LeftToRight`** (default): Carry propagates from MSB to LSB.
/// - **`RightToLeft`**: Carry propagates from LSB to MSB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CarryDirection {
    /// Carry propagates from left (MSB) to right (LSB).
    #[default]
    LeftToRight,
    /// Carry propagates from right (LSB) to left (MSB).
    RightToLeft,
}

/// Type alias for the standard LinearOperator used in this crate.
/// Uses TensorDynLen as the tensor type and usize as the node name type.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{
///     identity_mpo, tensortrain_to_linear_operator, QuanticsOperator,
/// };
/// let mpo = identity_mpo(1).unwrap();
/// let operator: QuanticsOperator = tensortrain_to_linear_operator(&mpo, &[2]).unwrap();
/// assert_eq!(operator.mpo().node_count(), 1);
/// ```
pub type QuanticsOperator = LinearOperator<TensorDynLen, usize>;

/// Convert a TensorTrain (MPO form) to a LinearOperator.
/// The TensorTrain is assumed to be an MPO with site dimension 4 (2x2 for input/output).
/// Each site tensor has shape (left_bond, site_dim=4, right_bond) where site_dim
/// encodes (s_out, s_in) = (2, 2).
/// # Arguments
/// * `tt` - TensorTrain representing an MPO
/// * `site_dims` - Site dimensions for input/output (typically all 2s)
/// # Returns
/// LinearOperator wrapping the MPO as a TreeTN.
/// # Errors
/// Returns an error when the tensor train cannot be converted to an operator
/// (a shape or index mismatch, or a backend failure).
/// # Examples
/// ```
/// use num_complex::Complex64;
/// use tensor4all_quanticstransform::tensortrain_to_linear_operator;
/// use tensor4all_simplett::{tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};
/// let mut tensor = tensor3_zeros(1, 4, 1);
/// tensor.set3(0, 0, 0, Complex64::new(1.0, 0.0));
/// tensor.set3(0, 3, 0, Complex64::new(1.0, 0.0));
/// let mpo = TensorTrain::new(vec![tensor]).unwrap();
/// let operator = tensortrain_to_linear_operator(&mpo, &[2]).unwrap();
/// assert_eq!(operator.mpo().node_count(), 1);
/// assert!(tensortrain_to_linear_operator(&mpo, &[2, 2]).is_err());
/// ```
pub fn tensortrain_to_linear_operator(
    tt: &TensorTrain<Complex64>,
    site_dims: &[usize],
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    let n = tt.len();
    if n == 0 {
        return Err(anyhow::anyhow!("Empty tensor train").into());
    }
    if site_dims.len() != n {
        return Err(anyhow::anyhow!("Dimension array must have length {n}").into());
    }
    let bond_capacity = n
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("tensor-train bond count overflows usize"))?;
    checked_allocation_len::<DynIndex>(&[n], "operator site indices")?;
    checked_allocation_len::<DynIndex>(&[bond_capacity], "operator bond indices")?;
    checked_allocation_len::<TensorDynLen>(&[n], "operator tensor list")?;
    checked_allocation_len::<usize>(&[n], "operator node-name list")?;
    for (site, &dim) in site_dims.iter().enumerate() {
        checked_allocation_len::<Complex64>(&[dim, dim], &format!("site {site}"))?;
    }

    // Create site indices for input and output
    let mut site_in_indices = try_vec_with_capacity::<DynIndex>("operator input site indices", n)?;
    let mut site_out_indices =
        try_vec_with_capacity::<DynIndex>("operator output site indices", n)?;
    let mut internal_in_indices =
        try_vec_with_capacity::<DynIndex>("operator internal input indices", n)?;
    let mut internal_out_indices =
        try_vec_with_capacity::<DynIndex>("operator internal output indices", n)?;

    for &dim in site_dims.iter() {
        // True site indices (for state)
        site_in_indices.push(Index::new_dyn(dim));
        site_out_indices.push(Index::new_dyn(dim));
        // Internal MPO indices
        internal_in_indices.push(Index::new_dyn(dim));
        internal_out_indices.push(Index::new_dyn(dim));
    }

    // Create bond indices
    let mut bond_indices =
        try_vec_with_capacity::<DynIndex>("operator bond indices", bond_capacity)?;

    for i in 0..=n {
        let dim = if i == 0 {
            1
        } else {
            tt.site_tensor(i - 1).right_dim()
        };
        bond_indices.push(Index::new_dyn(dim));
    }

    // Build tensors for TreeTN
    let mut tensors = try_vec_with_capacity::<TensorDynLen>("operator tensor list", n)?;
    let mut node_names = try_vec_with_capacity::<usize>("operator node-name list", n)?;

    for i in 0..n {
        let tensor = tt.site_tensor(i);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();

        // Expected site_dim is product of input and output dimensions
        let expected_site_dim = site_dims[i]
            .checked_mul(site_dims[i])
            .ok_or_else(|| anyhow::anyhow!("site {i} dimension product overflows usize"))?;
        if site_dim != expected_site_dim {
            return Err(anyhow::anyhow!(
                "Site {} has dimension {} but expected {} ({}x{})",
                i,
                site_dim,
                expected_site_dim,
                site_dims[i],
                site_dims[i]
            )
            .into());
        }

        // Create indices for this tensor: (left_bond, site_out, site_in, right_bond)
        // For first tensor: (site_out, site_in, right_bond)
        // For last tensor: (left_bond, site_out, site_in)
        // For middle: (left_bond, site_out, site_in, right_bond)
        let mut indices = try_vec_with_capacity::<DynIndex>("operator tensor indices", 4)?;
        let mut dims_vec = try_vec_with_capacity::<usize>("operator tensor dimensions", 4)?;

        if i > 0 {
            indices.push(bond_indices[i].clone());
            dims_vec.push(left_dim);
        }
        indices.push(internal_out_indices[i].clone());
        dims_vec.push(site_dims[i]);
        indices.push(internal_in_indices[i].clone());
        dims_vec.push(site_dims[i]);
        if i < n - 1 {
            indices.push(bond_indices[i + 1].clone());
            dims_vec.push(right_dim);
        }

        // Reshape tensor data: (left, site_out*site_in, right) -> (left, site_out, site_in, right)
        // or appropriate variant for boundary tensors
        let total_size = checked_allocation_len::<Complex64>(&dims_vec, &format!("site {i}"))?;
        let mut data = try_vec_with_capacity::<Complex64>("operator site data", total_size)?;
        data.resize(total_size, Complex64::new(0.0, 0.0));

        // Map from TT format to TreeTN format
        if i == 0 && n == 1 {
            // Single tensor: (site_out, site_in)
            for s_out in 0..site_dims[i] {
                for s_in in 0..site_dims[i] {
                    let s = s_out * site_dims[i] + s_in;
                    let idx = s_out + site_dims[i] * s_in;
                    data[idx] = *tensor.get3(0, s, 0);
                }
            }
        } else if i == 0 {
            // First tensor: (site_out, site_in, right_bond)
            for s_out in 0..site_dims[i] {
                for s_in in 0..site_dims[i] {
                    for r in 0..right_dim {
                        let s = s_out * site_dims[i] + s_in;
                        let idx = s_out + site_dims[i] * (s_in + site_dims[i] * r);
                        data[idx] = *tensor.get3(0, s, r);
                    }
                }
            }
        } else if i == n - 1 {
            // Last tensor: (left_bond, site_out, site_in)
            for l in 0..left_dim {
                for s_out in 0..site_dims[i] {
                    for s_in in 0..site_dims[i] {
                        let s = s_out * site_dims[i] + s_in;
                        let idx = l + left_dim * (s_out + site_dims[i] * s_in);
                        data[idx] = *tensor.get3(l, s, 0);
                    }
                }
            }
        } else {
            // Middle tensor: (left_bond, site_out, site_in, right_bond)
            for l in 0..left_dim {
                for s_out in 0..site_dims[i] {
                    for s_in in 0..site_dims[i] {
                        for r in 0..right_dim {
                            let s = s_out * site_dims[i] + s_in;
                            let idx =
                                l + left_dim * (s_out + site_dims[i] * (s_in + site_dims[i] * r));
                            data[idx] = *tensor.get3(l, s, r);
                        }
                    }
                }
            }
        }

        let tensor_dyn = TensorDynLen::from_dense(indices, data)?;
        tensors.push(tensor_dyn);
        node_names.push(i);
    }

    // Build TreeTN from tensors
    let treetn = TreeTN::from_tensors(tensors, node_names)?;

    // Build index mappings
    let mut input_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();

    for i in 0..n {
        input_mapping.insert(
            i,
            IndexMapping {
                true_index: site_in_indices[i].clone(),
                internal_index: internal_in_indices[i].clone(),
            },
        );
        output_mapping.insert(
            i,
            IndexMapping {
                true_index: site_out_indices[i].clone(),
                internal_index: internal_out_indices[i].clone(),
            },
        );
    }

    Ok(LinearOperator::new(treetn, input_mapping, output_mapping))
}

/// Convert a TensorTrain (MPO form) to a LinearOperator with asymmetric dimensions.
/// This variant supports different input and output dimensions, useful for
/// multi-variable transformations like affine transforms.
/// # Arguments
/// * `tt` - TensorTrain representing an MPO
/// * `input_dims` - Input dimensions per site
/// * `output_dims` - Output dimensions per site
/// # Returns
/// LinearOperator wrapping the MPO as a TreeTN.
/// # Errors
/// Returns an error when the tensor train cannot be converted to an operator
/// (a shape or index mismatch, or a backend failure).
/// # Examples
/// ```
/// use num_complex::Complex64;
/// use tensor4all_quanticstransform::tensortrain_to_linear_operator_asymmetric;
/// use tensor4all_simplett::{tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};
/// let mut tensor = tensor3_zeros(1, 6, 1);
/// tensor.set3(0, 0, 0, Complex64::new(1.0, 0.0));
/// tensor.set3(0, 5, 0, Complex64::new(1.0, 0.0));
/// let mpo = TensorTrain::new(vec![tensor]).unwrap();
/// let operator = tensortrain_to_linear_operator_asymmetric(&mpo, &[2], &[3]).unwrap();
/// assert_eq!(operator.mpo().node_count(), 1);
/// assert!(tensortrain_to_linear_operator_asymmetric(&mpo, &[2], &[2]).is_err());
/// ```
pub fn tensortrain_to_linear_operator_asymmetric(
    tt: &TensorTrain<Complex64>,
    input_dims: &[usize],
    output_dims: &[usize],
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    let n = tt.len();
    if n == 0 {
        return Err(anyhow::anyhow!("Empty tensor train").into());
    }
    if input_dims.len() != n || output_dims.len() != n {
        return Err(anyhow::anyhow!("Dimension arrays must have length {}", n).into());
    }
    let bond_capacity = n
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("tensor-train bond count overflows usize"))?;
    checked_allocation_len::<DynIndex>(&[n], "operator site indices")?;
    checked_allocation_len::<DynIndex>(&[bond_capacity], "operator bond indices")?;
    checked_allocation_len::<TensorDynLen>(&[n], "operator tensor list")?;
    checked_allocation_len::<usize>(&[n], "operator node-name list")?;
    for i in 0..n {
        checked_allocation_len::<Complex64>(
            &[output_dims[i], input_dims[i]],
            &format!("site {i}"),
        )?;
    }

    // Create site indices for input and output
    let mut site_in_indices = try_vec_with_capacity::<DynIndex>("operator input site indices", n)?;
    let mut site_out_indices =
        try_vec_with_capacity::<DynIndex>("operator output site indices", n)?;
    let mut internal_in_indices =
        try_vec_with_capacity::<DynIndex>("operator internal input indices", n)?;
    let mut internal_out_indices =
        try_vec_with_capacity::<DynIndex>("operator internal output indices", n)?;

    for i in 0..n {
        // True site indices (for state)
        site_in_indices.push(Index::new_dyn(input_dims[i]));
        site_out_indices.push(Index::new_dyn(output_dims[i]));
        // Internal MPO indices
        internal_in_indices.push(Index::new_dyn(input_dims[i]));
        internal_out_indices.push(Index::new_dyn(output_dims[i]));
    }

    // Create bond indices
    let mut bond_indices =
        try_vec_with_capacity::<DynIndex>("operator bond indices", bond_capacity)?;

    for i in 0..=n {
        let dim = if i == 0 {
            1
        } else {
            tt.site_tensor(i - 1).right_dim()
        };
        bond_indices.push(Index::new_dyn(dim));
    }

    // Build tensors for TreeTN
    let mut tensors = try_vec_with_capacity::<TensorDynLen>("operator tensor list", n)?;
    let mut node_names = try_vec_with_capacity::<usize>("operator node-name list", n)?;

    for i in 0..n {
        let tensor = tt.site_tensor(i);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();

        let in_dim = input_dims[i];
        let out_dim = output_dims[i];

        // Expected site_dim is product of input and output dimensions
        let expected_site_dim = in_dim
            .checked_mul(out_dim)
            .ok_or_else(|| anyhow::anyhow!("site {i} dimension product overflows usize"))?;
        if site_dim != expected_site_dim {
            return Err(anyhow::anyhow!(
                "Site {} has dimension {} but expected {} ({}x{})",
                i,
                site_dim,
                expected_site_dim,
                out_dim,
                in_dim
            )
            .into());
        }

        // Create indices for this tensor: (left_bond, site_out, site_in, right_bond)
        let mut indices = try_vec_with_capacity::<DynIndex>("operator tensor indices", 4)?;
        let mut dims_vec = try_vec_with_capacity::<usize>("operator tensor dimensions", 4)?;

        if i > 0 {
            indices.push(bond_indices[i].clone());
            dims_vec.push(left_dim);
        }
        indices.push(internal_out_indices[i].clone());
        dims_vec.push(out_dim);
        indices.push(internal_in_indices[i].clone());
        dims_vec.push(in_dim);
        if i < n - 1 {
            indices.push(bond_indices[i + 1].clone());
            dims_vec.push(right_dim);
        }

        // Reshape tensor data: (left, site_out*site_in, right) -> (left, site_out, site_in, right)
        let total_size = checked_allocation_len::<Complex64>(&dims_vec, &format!("site {i}"))?;
        let mut data = try_vec_with_capacity::<Complex64>("operator site data", total_size)?;
        data.resize(total_size, Complex64::new(0.0, 0.0));

        // Map from TT format to TreeTN format
        // TT format has site index = s_out * in_dim + s_in (output major, input minor)
        if i == 0 && n == 1 {
            // Single tensor: (site_out, site_in)
            for s_out in 0..out_dim {
                for s_in in 0..in_dim {
                    let s = s_out * in_dim + s_in;
                    let idx = s_out + out_dim * s_in;
                    data[idx] = *tensor.get3(0, s, 0);
                }
            }
        } else if i == 0 {
            // First tensor: (site_out, site_in, right_bond)
            for s_out in 0..out_dim {
                for s_in in 0..in_dim {
                    for r in 0..right_dim {
                        let s = s_out * in_dim + s_in;
                        let idx = s_out + out_dim * (s_in + in_dim * r);
                        data[idx] = *tensor.get3(0, s, r);
                    }
                }
            }
        } else if i == n - 1 {
            // Last tensor: (left_bond, site_out, site_in)
            for l in 0..left_dim {
                for s_out in 0..out_dim {
                    for s_in in 0..in_dim {
                        let s = s_out * in_dim + s_in;
                        let idx = l + left_dim * (s_out + out_dim * s_in);
                        data[idx] = *tensor.get3(l, s, 0);
                    }
                }
            }
        } else {
            // Middle tensor: (left_bond, site_out, site_in, right_bond)
            for l in 0..left_dim {
                for s_out in 0..out_dim {
                    for s_in in 0..in_dim {
                        for r in 0..right_dim {
                            let s = s_out * in_dim + s_in;
                            let idx = l + left_dim * (s_out + out_dim * (s_in + in_dim * r));
                            data[idx] = *tensor.get3(l, s, r);
                        }
                    }
                }
            }
        }

        let tensor_dyn = TensorDynLen::from_dense(indices, data)?;
        tensors.push(tensor_dyn);
        node_names.push(i);
    }

    // Build TreeTN from tensors
    let treetn = TreeTN::from_tensors(tensors, node_names)?;

    // Build index mappings
    let mut input_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();

    for i in 0..n {
        input_mapping.insert(
            i,
            IndexMapping {
                true_index: site_in_indices[i].clone(),
                internal_index: internal_in_indices[i].clone(),
            },
        );
        output_mapping.insert(
            i,
            IndexMapping {
                true_index: site_out_indices[i].clone(),
                internal_index: internal_out_indices[i].clone(),
            },
        );
    }

    Ok(LinearOperator::new(treetn, input_mapping, output_mapping))
}

pub(crate) fn checked_pow2(width: usize, name: &str) -> Result<usize> {
    let shift = u32::try_from(width)
        .map_err(|_| anyhow::anyhow!("{name} {width} exceeds usize shift width"))?;
    1usize
        .checked_shl(shift)
        .ok_or_else(|| anyhow::anyhow!("{name} {width} exceeds usize shift width"))
}

pub(crate) fn checked_allocation_len<T>(dims: &[usize], name: &str) -> Result<usize> {
    let elements = dims.iter().try_fold(1usize, |product, &dim| {
        product.checked_mul(dim).ok_or_else(|| {
            anyhow::anyhow!("{name} element count overflows usize for dimensions {dims:?}")
        })
    })?;
    let element_size = size_of::<T>();
    if element_size != 0 {
        let bytes = elements.checked_mul(element_size).ok_or_else(|| {
            anyhow::anyhow!("{name} byte length overflows usize for dimensions {dims:?}")
        })?;
        if bytes > isize::MAX as usize {
            return Err(anyhow::anyhow!(
                "{name} byte length exceeds isize::MAX for dimensions {dims:?}"
            ));
        }
    }
    Ok(elements)
}

pub(crate) fn try_vec_with_capacity<T>(name: &str, capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|err| {
        anyhow::anyhow!("{name} allocation failed for capacity {capacity}: {err}")
    })?;
    Ok(values)
}

pub(crate) fn checked_multivar_dims(nvariables: usize) -> Result<(usize, usize)> {
    if nvariables < 2 {
        anyhow::bail!("nvariables must be at least 2, got {nvariables}");
    }

    let local_dim = checked_pow2(nvariables, "nvariables")?;
    let site_dim = local_dim.checked_mul(local_dim).ok_or_else(|| {
        anyhow::anyhow!("multi-variable site dimension overflows usize for nvariables {nvariables}")
    })?;
    checked_allocation_len::<Complex64>(&[site_dim], "multi-variable site tensor")?;
    Ok((local_dim, site_dim))
}

/// Embed a single-variable MPO into a multi-variable context.
/// The original MPO acts on one variable (site_dim = d*d for d=2, i.e., in/out dim 2).
/// The embedded MPO acts on `nvariables` variables, applying the original
/// operator to `target_var` and identity on all others.
/// Site index encoding in the embedded MPO:
/// `s = s_out * (2^nvariables) + s_in` where
/// `s_out = var0_out + 2*var1_out + ...` and similarly for `s_in`.
/// # Arguments
/// * `mpo` - Single-variable MPO (R sites, site_dim = 4)
/// * `nvariables` - Total number of variables (must be >= 2)
/// * `target_var` - Which variable to apply the operator to (0-indexed)
pub(crate) fn embed_single_var_mpo(
    mpo: &TensorTrain<Complex64>,
    nvariables: usize,
    target_var: usize,
) -> Result<TensorTrain<Complex64>> {
    if target_var >= nvariables {
        return Err(anyhow::anyhow!(
            "target_var {} must be less than nvariables {}",
            target_var,
            nvariables
        ));
    }
    let (dim_multi, site_dim_new) = checked_multivar_dims(nvariables)?;
    let r = mpo.len();
    let mut new_tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "embedded MPO tensor list",
        r,
    )?;

    for i in 0..r {
        let tensor = mpo.site_tensor(i);
        let left_dim = tensor.left_dim();
        let right_dim = tensor.right_dim();

        assert_eq!(
            tensor.site_dim(),
            4,
            "Input MPO must have site_dim=4 (single variable)"
        );

        let total_size = checked_allocation_len::<Complex64>(
            &[left_dim, site_dim_new, right_dim],
            "embedded MPO tensor",
        )?;
        let mut data = try_vec_with_capacity::<Complex64>("embedded MPO tensor", total_size)?;
        data.resize(total_size, Complex64::new(0.0, 0.0));
        let mut t = tensor3_from_data(data, left_dim, site_dim_new, right_dim)
            .map_err(|err| anyhow::anyhow!("Failed to allocate embedded MPO tensor: {err}"))?;

        for s_out_multi in 0..dim_multi {
            for s_in_multi in 0..dim_multi {
                // Check identity constraint on non-target variables
                let mut identity_ok = true;
                for v in 0..nvariables {
                    if v != target_var {
                        let out_bit = (s_out_multi >> v) & 1;
                        let in_bit = (s_in_multi >> v) & 1;
                        if out_bit != in_bit {
                            identity_ok = false;
                            break;
                        }
                    }
                }
                if !identity_ok {
                    continue;
                }

                // Extract target variable bits
                let target_out = (s_out_multi >> target_var) & 1;
                let target_in = (s_in_multi >> target_var) & 1;
                let s_orig = target_out * 2 + target_in;

                // New fused site index
                let s_new = s_out_multi * dim_multi + s_in_multi;

                for l in 0..left_dim {
                    for rr in 0..right_dim {
                        let val = *tensor.get3(l, s_orig, rr);
                        if val != Complex64::new(0.0, 0.0) {
                            t.set3(l, s_new, rr, val);
                        }
                    }
                }
            }
        }

        new_tensors.push(t);
    }

    TensorTrain::new(new_tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create embedded MPO: {}", e))
}

/// Create an identity MPO for `r` binary sites.
/// # Errors
///
/// Returns an error when `r` is zero (an invalid-configuration failure) or
/// the site-list allocation overflows (an overflow failure).
/// # Examples
/// ```
/// use tensor4all_quanticstransform::identity_mpo;
/// use tensor4all_simplett::AbstractTensorTrain;
/// let mpo = identity_mpo(2).unwrap();
/// assert_eq!(mpo.len(), 2);
/// assert_eq!(mpo.site_dims(), vec![4, 4]);
/// assert!(identity_mpo(0).is_err());
/// ```
#[allow(dead_code)]
/// # Errors
/// Returns an error when the operator construction fails (an overflow or
/// invalid-configuration failure, or a shape mismatch).
///
pub fn identity_mpo(
    r: usize,
) -> std::result::Result<TensorTrain<Complex64>, QuanticsTransformError> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive").into());
    }
    let mut tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "identity MPO site list",
        r,
    )?;

    for _ in 0..r {
        // Identity tensor: delta_{s_out, s_in}
        // Shape: (1, 4, 1) where 4 = 2*2 for (s_out, s_in)
        let mut t = tensor3_zeros(1, 4, 1);
        // s = s_out * 2 + s_in
        // Identity: s_out == s_in
        t.set3(0, 0, 0, Complex64::one()); // (0, 0)
        t.set3(0, 3, 0, Complex64::one()); // (1, 1)
        tensors.push(t);
    }

    TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create identity MPO: {e}"))
        .map_err(QuanticsTransformError::from)
}

/// Create a scalar MPO (constant times identity).
#[allow(dead_code)]
/// # Errors
/// Returns an error when the operator construction fails (an overflow or
/// invalid-configuration failure, or a shape mismatch).
///
pub fn scalar_mpo(
    r: usize,
    value: Complex64,
) -> std::result::Result<TensorTrain<Complex64>, QuanticsTransformError> {
    let mut mpo = identity_mpo(r)?;
    mpo.scale(value);
    Ok(mpo)
}

#[cfg(test)]
mod tests;
