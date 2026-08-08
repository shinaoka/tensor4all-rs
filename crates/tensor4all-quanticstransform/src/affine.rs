//! Affine transformation operator: y = A*x + b
//!
//! This implements general affine transformations with rational coefficients.
//! The transformation computes y = A*x + b where A is an M×N rational matrix
//! and b is an M-dimensional rational vector.
//!
//! Based on the algorithm from Quantics.jl/src/affine.jl

use std::collections::HashMap;

use anyhow::Result;
use num_complex::Complex64;
use num_integer::Integer;
use num_rational::Rational64;
use num_traits::One;
use sprs::CsMat;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::LinearizationOrder;
use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};

use crate::common::{
    checked_pow2, tensortrain_to_linear_operator_asymmetric, BoundaryCondition, QuanticsOperator,
};
use tensor4all_simplett::tensor::Tensor3 as GenericTensor3;

#[derive(Clone, Debug)]
struct BoolTensor<const N: usize> {
    data: Vec<u8>,
    dims: [usize; N],
}

type BoolTensor2 = BoolTensor<2>;
type BoolTensor3 = BoolTensor<3>;

impl<const N: usize> BoolTensor<N> {
    fn from_elem(dims: [usize; N], value: bool) -> Self {
        let total: usize = dims.iter().product();
        Self {
            data: vec![u8::from(value); total],
            dims,
        }
    }

    fn dims(&self) -> &[usize; N] {
        &self.dims
    }

    fn get(&self, idx: [usize; N]) -> bool {
        self.data[self.offset(&idx)] != 0
    }

    fn set(&mut self, idx: [usize; N], value: bool) {
        let offset = self.offset(&idx);
        self.data[offset] = u8::from(value);
    }

    fn offset(&self, idx: &[usize; N]) -> usize {
        let mut stride = 1usize;
        let mut offset = 0usize;
        for axis in (0..N).rev() {
            offset += idx[axis] * stride;
            stride *= self.dims[axis];
        }
        offset
    }
}

/// A primitive integer row for a linear equality or inequality constraint.
///
/// Use this type when a row is scale-invariant, such as `a*x == rhs` or
/// `a*x <= rhs`, and the row will be used to derive affine or halfspace
/// transform operators. It is intentionally separate from [`AffineParams`]
/// because an affine map `y = A*x + b` is not invariant under row scaling.
///
/// Related types: [`AffineParams`] stores affine-map parameters for
/// [`affine_operator`]; this type stores normalized constraint rows that can be
/// used before constructing a constraint-derived operator.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::LinearConstraintRow;
///
/// let row = LinearConstraintRow::from_integers(vec![16], 64);
/// assert_eq!(row.coefficients, vec![1]);
/// assert_eq!(row.rhs, 4);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearConstraintRow {
    /// Integer coefficients in primitive form.
    ///
    /// The row is normalized by clearing rational denominators when needed and
    /// then dividing all coefficients and the right-hand side by their positive
    /// greatest common divisor. Use these coefficients when a constraint row is
    /// scale-invariant, such as `a*x <= rhs` or `a*x == rhs`.
    pub coefficients: Vec<i64>,
    /// Integer right-hand side in primitive form.
    ///
    /// This value is reduced with [`Self::coefficients`]. For example,
    /// `16*x <= 64` is represented as `coefficients = [1]` and `rhs = 4`.
    pub rhs: i64,
}

impl LinearConstraintRow {
    /// Create a primitive integer constraint row.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Coefficients of the left-hand side. The entries may
    ///   share a positive common factor with `rhs`; that factor is removed.
    /// * `rhs` - Right-hand side of the equality or inequality constraint.
    ///
    /// # Returns
    ///
    /// A row with the same represented equality or inequality set under
    /// positive scaling. If all coefficients and `rhs` are zero, the zero row
    /// is returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::LinearConstraintRow;
    ///
    /// let row = LinearConstraintRow::from_integers(vec![16], 64);
    /// assert_eq!(row.coefficients, vec![1]);
    /// assert_eq!(row.rhs, 4);
    ///
    /// let negative = LinearConstraintRow::from_integers(vec![-16], -64);
    /// assert_eq!(negative.coefficients, vec![-1]);
    /// assert_eq!(negative.rhs, -4);
    /// ```
    pub fn from_integers(coefficients: Vec<i64>, rhs: i64) -> Self {
        let common_factor = coefficients
            .iter()
            .chain(std::iter::once(&rhs))
            .fold(0i64, |factor, value| factor.gcd(value))
            .abs();

        if common_factor > 1 {
            Self {
                coefficients: coefficients
                    .into_iter()
                    .map(|coefficient| coefficient / common_factor)
                    .collect(),
                rhs: rhs / common_factor,
            }
        } else {
            Self { coefficients, rhs }
        }
    }

    /// Create a primitive constraint row from rational values.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Rational coefficients of the left-hand side. The
    ///   least common multiple of all denominators is used to clear
    ///   denominators before gcd reduction.
    /// * `rhs` - Rational right-hand side of the equality or inequality
    ///   constraint.
    ///
    /// # Returns
    ///
    /// A primitive integer row equivalent to the rational constraint under
    /// positive scaling. Use this for constraint rows before deriving
    /// affine/halfspace projector operators; do not use it to simplify a
    /// general affine map `y = A*x + b`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_rational::Rational64;
    /// use tensor4all_quanticstransform::LinearConstraintRow;
    ///
    /// let row = LinearConstraintRow::from_rationals(
    ///     vec![Rational64::new(2, 3), Rational64::new(4, 3)],
    ///     Rational64::from_integer(2),
    /// );
    /// assert_eq!(row.coefficients, vec![1, 2]);
    /// assert_eq!(row.rhs, 3);
    /// ```
    pub fn from_rationals(coefficients: Vec<Rational64>, rhs: Rational64) -> Self {
        let mut denominator_lcm = 1i64;
        for coefficient in &coefficients {
            denominator_lcm = denominator_lcm.lcm(coefficient.denom());
        }
        denominator_lcm = denominator_lcm.lcm(rhs.denom());

        let integer_coefficients = coefficients
            .iter()
            .map(|coefficient| (coefficient * denominator_lcm).to_integer())
            .collect();
        let integer_rhs = (rhs * denominator_lcm).to_integer();

        Self::from_integers(integer_coefficients, integer_rhs)
    }
}

/// Affine transformation parameters.
///
/// Represents the transformation y = A*x + b where:
/// - A is an M x N matrix stored in column-major order
/// - b is an M-dimensional vector
/// - x is an N-dimensional input
/// - y is an M-dimensional output
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::AffineParams;
/// use num_rational::Rational64;
///
/// // 1D shift: y = x + 3
/// let params = AffineParams::from_integers(vec![1], vec![3], 1, 1).unwrap();
/// assert_eq!(params.m, 1);
/// assert_eq!(params.n, 1);
///
/// // 2D rotation: y = [[1,1],[1,-1]] * x
/// // Column-major: [A[0,0], A[1,0], A[0,1], A[1,1]]
/// let params = AffineParams::from_integers(
///     vec![1, 1, 1, -1], vec![0, 0], 2, 2
/// ).unwrap();
/// assert_eq!(params.m, 2);
/// assert_eq!(params.n, 2);
///
/// // With rational coefficients: y = (1/2)*x
/// let params = AffineParams::new(
///     vec![Rational64::new(1, 2)],
///     vec![Rational64::from_integer(0)],
///     1, 1,
/// ).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct AffineParams {
    /// Transformation matrix A (M×N), stored in column-major order
    pub a: Vec<Rational64>,
    /// Translation vector b (M elements)
    pub b: Vec<Rational64>,
    /// Number of output dimensions (M)
    pub m: usize,
    /// Number of input dimensions (N)
    pub n: usize,
}

impl AffineParams {
    /// Create new affine parameters.
    ///
    /// # Arguments
    /// * `a` - M x N matrix in column-major order (length must be m*n)
    /// * `b` - M-dimensional translation vector (length must be m)
    /// * `m` - Number of output dimensions
    /// * `n` - Number of input dimensions
    ///
    /// # Errors
    /// Returns an error when `a.len() != m * n` or `b.len() != m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::AffineParams;
    /// use num_rational::Rational64;
    ///
    /// // 1D identity: y = x
    /// let params = AffineParams::new(
    ///     vec![Rational64::from_integer(1)],
    ///     vec![Rational64::from_integer(0)],
    ///     1, 1,
    /// ).unwrap();
    ///
    /// // Dimension mismatch errors
    /// assert!(AffineParams::new(
    ///     vec![Rational64::from_integer(1)],
    ///     vec![Rational64::from_integer(0)],
    ///     2, 1, // expects 2 elements in A, got 1
    /// ).is_err());
    /// ```
    pub fn new(a: Vec<Rational64>, b: Vec<Rational64>, m: usize, n: usize) -> Result<Self> {
        let expected_a_len = m
            .checked_mul(n)
            .ok_or_else(|| anyhow::anyhow!("Affine matrix dimensions overflow usize: {m} × {n}"))?;
        checked_pow2(m, "output variable count")?;
        checked_pow2(n, "input variable count")?;
        if a.len() != expected_a_len {
            return Err(anyhow::anyhow!(
                "Matrix A has {} elements but expected {}×{}={}",
                a.len(),
                m,
                n,
                expected_a_len
            ));
        }
        if b.len() != m {
            return Err(anyhow::anyhow!(
                "Vector b has {} elements but expected {}",
                b.len(),
                m
            ));
        }
        Ok(Self { a, b, m, n })
    }

    /// Create affine parameters from integer matrix and vector.
    ///
    /// Convenience method that converts integer values to rationals.
    ///
    /// # Errors
    /// Returns an error when `a.len() != m * n` or `b.len() != m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::AffineParams;
    ///
    /// // 2D: y = [[1, 0], [0, 1]] * x + [1, 2] (shift by (1,2))
    /// let params = AffineParams::from_integers(
    ///     vec![1, 0, 0, 1], vec![1, 2], 2, 2,
    /// ).unwrap();
    /// assert_eq!(params.m, 2);
    /// assert_eq!(params.n, 2);
    /// ```
    pub fn from_integers(a: Vec<i64>, b: Vec<i64>, m: usize, n: usize) -> Result<Self> {
        let a_rat: Vec<Rational64> = a.into_iter().map(Rational64::from_integer).collect();
        let b_rat: Vec<Rational64> = b.into_iter().map(Rational64::from_integer).collect();
        Self::new(a_rat, b_rat, m, n)
    }

    /// Get element A[i, j] (0-indexed)
    #[allow(dead_code)]
    fn get_a(&self, i: usize, j: usize) -> Rational64 {
        self.a[i + self.m * j]
    }

    /// Convert to integer representation by scaling with LCM of denominators.
    /// Returns (A_int, b_int, scale) where A_int = scale * A and b_int = scale * b.
    fn to_integer_scaled(&self) -> (Vec<i64>, Vec<i64>, i64) {
        // Find LCM of all denominators
        let mut denom_lcm = 1i64;
        for r in &self.a {
            denom_lcm = denom_lcm.lcm(r.denom());
        }
        for r in &self.b {
            denom_lcm = denom_lcm.lcm(r.denom());
        }

        // Scale to integers
        let a_int: Vec<i64> = self
            .a
            .iter()
            .map(|r| (r * denom_lcm).to_integer())
            .collect();
        let b_int: Vec<i64> = self
            .b
            .iter()
            .map(|r| (r * denom_lcm).to_integer())
            .collect();

        (a_int, b_int, denom_lcm)
    }
}

fn affine_boundary_weight(carry: &[i64], bc: &[BoundaryCondition]) -> f64 {
    carry
        .iter()
        .zip(bc.iter())
        .map(|(&c, &boundary)| match boundary {
            BoundaryCondition::Periodic => 1.0,
            BoundaryCondition::AntiPeriodic => {
                if c.rem_euclid(2) == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            BoundaryCondition::Open => {
                if c == 0 {
                    1.0
                } else {
                    0.0
                }
            }
        })
        .product()
}

fn affine_needs_extension(bc: &[BoundaryCondition], b_work: &[i64]) -> bool {
    b_work.iter().any(|&b| b > 0)
        && bc
            .iter()
            .any(|b| matches!(b, BoundaryCondition::AntiPeriodic | BoundaryCondition::Open))
}

/// Remap site indices of the affine MPO from internal encoding to the convention
/// expected by `tensortrain_to_linear_operator_asymmetric`.
///
/// Internal encoding: `site_idx = y_bits | (x_bits << m)` (y-minor, x-major)
/// Expected encoding: `s = s_out * in_dim + s_in = y_bits * 2^n + x_bits` (x-minor, y-major)
fn checked_affine_site_dims(m: usize, n: usize) -> Result<(usize, usize, usize)> {
    let input_dim = checked_pow2(n, "input variable count")?;
    let output_dim = checked_pow2(m, "output variable count")?;
    let site_dim = input_dim.checked_mul(output_dim).ok_or_else(|| {
        anyhow::anyhow!(
            "affine site dimension overflows usize for {m} output variables and {n} input variables"
        )
    })?;
    Ok((input_dim, output_dim, site_dim))
}

fn remap_affine_site_indices(
    mpo: &TensorTrain<Complex64>,
    m: usize,
    n: usize,
    site_dim: usize,
) -> Result<TensorTrain<Complex64>> {
    let input_dim = checked_pow2(n, "input variable count")?;
    let output_dim = checked_pow2(m, "output variable count")?;

    // Build permutation table: perm[old_idx] = remapped index
    let perm: Vec<usize> = (0..site_dim)
        .map(|old_idx| {
            let y_bits = old_idx & (output_dim - 1);
            let x_bits = old_idx >> m;
            y_bits * input_dim + x_bits
        })
        .collect();

    let r = mpo.len();
    let mut new_tensors = Vec::with_capacity(r);

    for i in 0..r {
        let tensor = mpo.site_tensor(i);
        let left_dim = tensor.left_dim();
        let right_dim = tensor.right_dim();

        let mut t = tensor3_zeros(left_dim, site_dim, right_dim);
        for l in 0..left_dim {
            for (old_s, &new_s) in perm.iter().enumerate() {
                for rr in 0..right_dim {
                    let val = *tensor.get3(l, old_s, rr);
                    if val != Complex64::new(0.0, 0.0) {
                        t.set3(l, new_s, rr, val);
                    }
                }
            }
        }
        new_tensors.push(t);
    }

    TensorTrain::new(new_tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create remapped MPO: {}", e))
}

/// Create the operator that realizes the coordinate map `y = A * x + b`.
///
/// This is the **forward** affine operator. It maps a quantics tensor train
/// representing an `N`-variable state `x` to the quantics tensor train of
/// the `M`-variable state `y = A * x + b`.
///
/// To build the **pullback** (`f(y) = g(A * y + b)`), call `.transpose()`
/// on the returned operator; the pullback is exactly the transpose of the
/// forward operator.
///
/// # Arguments
///
/// * `r` — bits per variable (number of sites in the output MPO).
/// * `params` — rational `M × N` matrix `A` and `M`-vector `b` describing
///   the affine map.
/// * `bc` — length `M` array of boundary conditions for each output variable.
///   `Periodic` wraps output coordinates modulo `2^r`; `Open` zeroes the
///   out-of-range contributions.
///
/// # Errors
///
/// Returns an error if `r == 0` or if `bc.len() != params.m`.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{affine_operator, AffineParams, BoundaryCondition};
/// use num_rational::Rational64;
///
/// // Transform g(x, y) -> g(x + y, x - y) (rotation by 45 degrees, scaled)
/// let a = vec![
///     Rational64::from_integer(1), Rational64::from_integer(1),  // row 0: x + y
///     Rational64::from_integer(1), Rational64::from_integer(-1), // row 1: x - y
/// ];
/// let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
/// let params = AffineParams::new(a, b, 2, 2).unwrap();
/// let bc = vec![BoundaryCondition::Periodic; 2];
/// let op = affine_operator(4, &params, &bc).unwrap();
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
///
/// Using integer convenience constructor:
///
/// ```
/// use tensor4all_quanticstransform::{affine_operator, AffineParams, BoundaryCondition};
///
/// // Identity transform: y = x (1D)
/// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
/// let bc = vec![BoundaryCondition::Periodic];
/// let op = affine_operator(4, &params, &bc).unwrap();
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn affine_operator(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }
    if bc.len() != params.m {
        return Err(anyhow::anyhow!(
            "Boundary conditions length {} doesn't match output dimensions {}",
            bc.len(),
            params.m
        ));
    }

    // Site dimensions: M output variables, N input variables
    // Input dimension per site: 2^N (N input bits)
    // Output dimension per site: 2^M (M output bits)
    let m = params.m;
    let n = params.n;
    let (input_dim, output_dim, site_dim) = checked_affine_site_dims(m, n)?;

    let mpo = affine_transform_mpo(r, params, bc)?;

    // The internal affine MPO uses site encoding: site_idx = y_bits | (x_bits << m)
    // (y-minor, x-major). But tensortrain_to_linear_operator_asymmetric expects
    // s = s_out * in_dim + s_in = y_bits * 2^N + x_bits (x-minor, y-major).
    // We need to remap the site indices.
    let remapped_mpo = remap_affine_site_indices(&mpo, m, n, site_dim)?;

    let input_dims = vec![input_dim; r];
    let output_dims = vec![output_dim; r];
    tensortrain_to_linear_operator_asymmetric(&remapped_mpo, &input_dims, &output_dims)
}

/// Create an affine operator with interleaved binary variable indices.
///
/// This is the same forward coordinate map as [`affine_operator`], but each bit
/// node carries one binary output index per output variable and one binary input
/// index per input variable instead of fusing variables into local dimensions
/// `2^M` and `2^N`. The mapping order at each node is
/// `(y0, y1, ..., yM-1)` for outputs and `(x0, x1, ..., xN-1)` for inputs.
///
/// Use this form when the state stores variables as separate interleaved QTT
/// site indices and should bind them through [`LinearOperator::new_multi`].
///
/// # Arguments
///
/// * `r` - Bits per variable. Node `0` is the most significant bit.
/// * `params` - Rational affine map `y = A*x + b`.
/// * `bc` - Boundary condition for each output variable.
///
/// # Returns
///
/// A [`LinearOperator`] whose node `i` has `params.n` input mappings and
/// `params.m` output mappings, all with binary dimension.
///
/// # Errors
///
/// Returns an error when `r == 0`, when `bc.len() != params.m`, or when the
/// affine tensor network cannot be constructed.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{
///     affine_operator_interleaved, AffineParams, BoundaryCondition,
/// };
///
/// let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
/// let bc = vec![BoundaryCondition::Periodic; 2];
/// let op = affine_operator_interleaved(3, &params, &bc).unwrap();
///
/// assert_eq!(op.mpo().node_count(), 3);
/// assert_eq!(op.get_output_mappings(&0).unwrap().len(), 2);
/// assert_eq!(op.get_input_mappings(&0).unwrap().len(), 2);
/// ```
pub fn affine_operator_interleaved(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    let mut op = affine_operator(r, params, bc)?;

    let fused_output_indices = (0..r)
        .map(|site| {
            op.get_output_mapping(&site)
                .ok_or_else(|| anyhow::anyhow!("missing affine output mapping for site {site}"))
                .map(|mapping| mapping.true_index.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    let fused_input_indices = (0..r)
        .map(|site| {
            op.get_input_mapping(&site)
                .ok_or_else(|| anyhow::anyhow!("missing affine input mapping for site {site}"))
                .map(|mapping| mapping.true_index.clone())
        })
        .collect::<Result<Vec<_>>>()?;

    for site in 0..r {
        let output_indices = (0..params.m)
            .map(|_| Index::<DynId, TagSet>::new_dyn(2))
            .collect::<Vec<_>>();
        op = op.unfuse_output_index(
            &fused_output_indices[site],
            &output_indices,
            LinearizationOrder::ColumnMajor,
        )?;

        let input_indices = (0..params.n)
            .map(|_| Index::<DynId, TagSet>::new_dyn(2))
            .collect::<Vec<_>>();
        op = op.unfuse_input_index(
            &fused_input_indices[site],
            &input_indices,
            LinearizationOrder::ColumnMajor,
        )?;
    }

    Ok(op)
}

/// Compute the full affine transformation matrix directly (for verification).
///
/// This computes the transformation matrix by directly evaluating y = A*x + b
/// for all possible input values. The result is a sparse boolean matrix.
///
/// # Arguments
/// * `r` - Number of bits per variable
/// * `params` - Affine transformation parameters
/// * `bc` - Boundary conditions for each output variable
///
/// # Returns
/// Sparse matrix of size 2^(R*M) × 2^(R*N) where entry (y_flat, x_flat) = 1
/// if the transformation maps x to y.
///
/// # Note
/// This is only practical for small R due to exponential size.
/// Use for testing/verification only.
///
/// # Errors
/// Returns an error when `r == 0` or when `bc.len() != params.m`.
pub fn affine_transform_matrix(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<CsMat<f64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }
    if bc.len() != params.m {
        return Err(anyhow::anyhow!(
            "Boundary conditions length {} doesn't match output dimensions {}",
            bc.len(),
            params.m
        ));
    }

    let (a_int, b_int, scale) = params.to_integer_scaled();
    let m = params.m;
    let n = params.n;

    let input_exponent = r.checked_mul(n).ok_or_else(|| {
        anyhow::anyhow!("affine input dimension exponent overflows usize: {r} * {n}")
    })?;
    let output_exponent = r.checked_mul(m).ok_or_else(|| {
        anyhow::anyhow!("affine output dimension exponent overflows usize: {r} * {m}")
    })?;
    let input_size = checked_pow2(input_exponent, "affine input dimension exponent")?;
    let output_size = checked_pow2(output_exponent, "affine output dimension exponent")?;
    let modulus_shift = u32::try_from(r)
        .map_err(|_| anyhow::anyhow!("number of bits {r} exceeds signed modulus width"))?;
    let modulus = 1i64
        .checked_shl(modulus_shift)
        .ok_or_else(|| anyhow::anyhow!("number of bits {r} exceeds signed modulus width"))?;
    if r >= (i64::BITS - 1) as usize {
        anyhow::bail!("number of bits {r} exceeds signed modulus width");
    }
    let bit_mask = checked_pow2(r, "number of bits")? - 1;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    // Iterate over all (x, y) pairs, matching Julia's approach.
    // For periodic BC with scale > 1, multiple y values can satisfy
    // scale * y ≡ A*x + b (mod 2^R), so we must check all pairs.
    for x_flat in 0..input_size {
        // Decode x_flat to N-dimensional x vector
        // x_flat = x[0] + x[1]*2^R + x[2]*2^(2R) + ...
        let x: Vec<i64> = (0..n)
            .map(|var| ((x_flat >> (var * r)) & bit_mask) as i64)
            .collect();

        // Compute v = A*x + b (unscaled)
        let mut v: Vec<i64> = vec![0; m];
        for i in 0..m {
            v[i] = b_int[i];
            for j in 0..n {
                v[i] += a_int[i + m * j] * x[j];
            }
        }

        for y_flat in 0..output_size {
            // Decode y_flat to M-dimensional y vector
            let y: Vec<i64> = (0..m)
                .map(|var| ((y_flat >> (var * r)) & bit_mask) as i64)
                .collect();

            let mut carry = vec![0i64; m];
            let mut valid = true;
            for i in 0..m {
                let diff = v[i] - scale * y[i];
                match bc[i] {
                    BoundaryCondition::Periodic | BoundaryCondition::AntiPeriodic => {
                        if diff.rem_euclid(modulus) == 0 {
                            carry[i] = diff / modulus;
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    BoundaryCondition::Open => {
                        if diff == 0 {
                            carry[i] = 0;
                        } else {
                            valid = false;
                            break;
                        }
                    }
                }
            }

            if valid {
                let weight = affine_boundary_weight(&carry, bc);
                if weight == 0.0 {
                    continue;
                }
                rows.push(y_flat);
                cols.push(x_flat);
                vals.push(weight);
            }
        }
    }

    // Build sparse matrix in CSR format
    let triplet = sprs::TriMat::from_triplets((output_size, input_size), rows, cols, vals);
    Ok(triplet.to_csr())
}

/// Create the affine transformation MPO as a TensorTrain.
fn affine_transform_mpo(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<TensorTrain<Complex64>> {
    let (a_int, b_int, scale) = params.to_integer_scaled();
    let m = params.m;
    let n = params.n;

    // Compute core tensors
    let tensors = affine_transform_tensors(r, &a_int, &b_int, scale, m, n, bc)?;

    TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create affine transform MPO: {}", e))
}

/// Create unfused affine transformation tensors.
///
/// Returns a vector of R tensors, where each tensor has shape:
/// `[left_bond, 2, 2, ..., 2, right_bond]` with M+N physical indices of dimension 2.
///
/// The physical index order matches Quantics.jl:
/// `(y[1], y[2], ..., y[M], x[1], x[2], ..., x[N])`
/// where y are output variables and x are input variables.
///
/// # Arguments
/// * `r` - Number of bits per variable (number of sites)
/// * `params` - Affine transformation parameters
/// * `bc` - Boundary conditions for each output variable
///
/// # Returns
/// Vector of R tensors with unfused physical indices.
///
/// # Errors
/// Returns an error when `r == 0`, when `bc.len() != params.m`, or when affine
/// tensor construction fails.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{
///     affine_transform_tensors_unfused, AffineParams, BoundaryCondition,
/// };
/// use tensor4all_simplett::Tensor3Ops;
///
/// let params = AffineParams::from_integers(vec![1, 1, 0, 1], vec![0, 0], 2, 2).unwrap();
/// let bc = vec![BoundaryCondition::Periodic; 2];
/// let tensors = affine_transform_tensors_unfused(4, &params, &bc).unwrap();
///
/// // One tensor per site
/// assert_eq!(tensors.len(), 4);
///
/// // Each tensor has fused site_dim = 2^(M+N) = 16 for M=2, N=2
/// assert_eq!(tensors[0].site_dim(), 16);
/// ```
pub fn affine_transform_tensors_unfused(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<Vec<GenericTensor3<Complex64>>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }
    if bc.len() != params.m {
        return Err(anyhow::anyhow!(
            "Boundary conditions length {} doesn't match output dimensions {}",
            bc.len(),
            params.m
        ));
    }

    let (a_int, b_int, scale) = params.to_integer_scaled();
    let m = params.m;
    let n = params.n;
    let (_, _, site_dim) = checked_affine_site_dims(m, n)?;

    // Compute fused tensors first
    let fused_tensors = affine_transform_tensors(r, &a_int, &b_int, scale, m, n, bc)?;

    // Convert fused tensors to unfused format
    // Fused: [left, fused_site, right] where fused_site = 2^(M+N)
    // Unfused: [left, 2, 2, ..., 2, right] with M+N dimensions of size 2
    //
    // Fused index encoding: site_idx = y_bits | (x_bits << M)
    // where y_bits = y[0] + 2*y[1] + ... + 2^(M-1)*y[M-1]
    // and   x_bits = x[0] + 2*x[1] + ... + 2^(N-1)*x[N-1]
    //
    // Quantics.jl order: (y[0], y[1], ..., y[M-1], x[0], x[1], ..., x[N-1])
    // We preserve that semantic index order:
    // unfused[left, y0, y1, ..., yM-1, x0, x1, ..., xN-1, right]

    let mut unfused_tensors = Vec::with_capacity(r);

    for tensor in fused_tensors.iter() {
        let left_dim = tensor.left_dim();
        let right_dim = tensor.right_dim();

        // Create unfused tensor
        // Shape: [left_dim, 2^(M+N), right_dim] but we keep it as 3D for now
        // The reshape to (M+N+2)-dimensional tensor will be done by the caller if needed
        // For now, we provide a 3D tensor where the middle dimension is the fused site
        // and document how to unfuse it.
        //
        // Actually, let's return it properly unfused using a flat storage with
        // the correct index order for reshape.
        //
        // Total size: left_dim * 2^(M+N) * right_dim
        // Shape for unfused: [left_dim, 2, 2, ..., 2, right_dim]
        //
        // Index mapping from fused to unfused:
        // fused site_idx -> (y0, y1, ..., yM-1, x0, x1, ..., xN-1)
        // site_idx = y0 + 2*y1 + ... + 2^(M-1)*yM-1 + 2^M * (x0 + 2*x1 + ...)

        // Preserve the Quantics.jl physical index order
        // (y0, y1, ..., yM-1, x0, x1, ..., xN-1).

        let mut unfused_data = vec![Complex64::new(0.0, 0.0); left_dim * site_dim * right_dim];

        for l in 0..left_dim {
            for fused_idx in 0..site_dim {
                for rr in 0..right_dim {
                    let val = tensor.get3(l, fused_idx, rr);
                    if val.norm() > 0.0 {
                        // The fused index encodes: site_idx = y_bits | (x_bits << M)
                        // This matches Quantics.jl's ordering (y variables first, then x)
                        // so we can use fused_idx directly.
                        //
                        // The caller can reshape [left, site_dim, right] to
                        // [left, 2, 2, ..., 2, right] with M+N dimensions of size 2,
                        // where indices are in order (y[0], y[1], ..., y[M-1], x[0], ..., x[N-1])
                        let flat_idx = l * site_dim * right_dim + fused_idx * right_dim + rr;
                        unfused_data[flat_idx] = *val;
                    }
                }
            }
        }

        // Create Tensor3 with shape [left_dim, site_dim, right_dim]
        // The caller can reshape this to [left_dim, 2, 2, ..., 2, right_dim]
        // with the understanding that the indices are ordered as (y0, y1, ..., x0, x1, ...)
        let mut unfused_tensor =
            GenericTensor3::from_elem([left_dim, site_dim, right_dim], Complex64::new(0.0, 0.0));
        for l in 0..left_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    unfused_tensor[[l, s, r]] =
                        unfused_data[l * site_dim * right_dim + s * right_dim + r];
                }
            }
        }

        unfused_tensors.push(unfused_tensor);
    }

    Ok(unfused_tensors)
}

/// Information about the unfused tensor structure.
///
/// This helper provides metadata for reshaping the unfused tensors
/// produced by [`affine_transform_tensors_unfused`].
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
///
/// let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
/// let info = UnfusedTensorInfo::new(&params);
///
/// assert_eq!(info.m, 2);
/// assert_eq!(info.n, 2);
/// assert_eq!(info.num_physical_dims, 4);
///
/// // Get shape for a tensor with bond dims 3 and 5
/// let shape = info.unfused_shape(3, 5);
/// assert_eq!(shape, vec![3, 2, 2, 2, 2, 5]);
///
/// // Round-trip encode/decode
/// let fused = info.encode_fused_index(&[1, 0], &[0, 1]);
/// let (y_bits, x_bits) = info.decode_fused_index(fused);
/// assert_eq!(y_bits, vec![1, 0]);
/// assert_eq!(x_bits, vec![0, 1]);
/// ```
#[derive(Clone, Debug)]
pub struct UnfusedTensorInfo {
    /// Number of output variables (M)
    pub m: usize,
    /// Number of input variables (N)
    pub n: usize,
    /// Total physical dimensions per site (M + N)
    pub num_physical_dims: usize,
    /// Dimension of each physical index (always 2)
    pub physical_dim: usize,
}

impl UnfusedTensorInfo {
    /// Create info for the given affine parameters.
    pub fn new(params: &AffineParams) -> Self {
        Self {
            m: params.m,
            n: params.n,
            num_physical_dims: params.m + params.n,
            physical_dim: 2,
        }
    }

    /// Get the shape for a fully unfused tensor at a given site.
    ///
    /// Returns `[left_bond, 2, 2, ..., 2, right_bond]` where there are M+N 2s.
    pub fn unfused_shape(&self, left_bond: usize, right_bond: usize) -> Vec<usize> {
        let mut shape = Vec::with_capacity(2 + self.num_physical_dims);
        shape.push(left_bond);
        shape.extend(std::iter::repeat_n(2, self.num_physical_dims));
        shape.push(right_bond);
        shape
    }

    /// Decode a fused site index to individual variable bits.
    ///
    /// Returns `(y_bits, x_bits)` where:
    /// - `y_bits[i]` is the bit for output variable i
    /// - `x_bits[j]` is the bit for input variable j
    pub fn decode_fused_index(&self, fused_idx: usize) -> (Vec<usize>, Vec<usize>) {
        let y_combined = fused_idx & ((1 << self.m) - 1);
        let x_combined = fused_idx >> self.m;

        let y_bits: Vec<usize> = (0..self.m).map(|i| (y_combined >> i) & 1).collect();
        let x_bits: Vec<usize> = (0..self.n).map(|j| (x_combined >> j) & 1).collect();

        (y_bits, x_bits)
    }

    /// Encode individual variable bits to a fused site index.
    ///
    /// # Arguments
    /// * `y_bits` - Bits for output variables (length M)
    /// * `x_bits` - Bits for input variables (length N)
    pub fn encode_fused_index(&self, y_bits: &[usize], x_bits: &[usize]) -> usize {
        let y_combined: usize = y_bits.iter().enumerate().map(|(i, &b)| b << i).sum();
        let x_combined: usize = x_bits.iter().enumerate().map(|(j, &b)| b << j).sum();
        y_combined | (x_combined << self.m)
    }
}

/// Compute the core tensors for the affine transformation.
///
/// This implements the algorithm from Quantics.jl that handles:
/// - Carry propagation for multi-bit arithmetic
/// - Scaling factor s from rational to integer conversion
///
/// Uses big-endian convention: site 0 = MSB, site R-1 = LSB.
///
/// Carry propagation direction (matching shift.rs):
/// - Arithmetic carry flows LSB → MSB (physical fact)
/// - In big-endian: site R-1 → site 0 (right → left)
/// - Tensor structure: t[left, site, right] where left=carry_out (going left), right=carry_in (from right)
/// - Site 0 (MSB): BC applied on left, receives carry from right → shape (1, site_dim, num_carries)
/// - Site R-1 (LSB): initial carry=0, sends carry to left → shape (num_carries, site_dim, 1)
/// - Middle sites: shape (num_carries, site_dim, num_carries)
fn affine_transform_tensors(
    r: usize,
    a_int: &[i64],
    b_int: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    bc: &[BoundaryCondition],
) -> Result<Vec<tensor4all_simplett::Tensor3<Complex64>>> {
    let (_, _, site_dim) = checked_affine_site_dims(m, n)?;

    // Track sign separately and work with absolute value
    // so that right-shifting always terminates (Julia PR #45 approach)
    let bsign: Vec<i64> = b_int.iter().map(|&b| if b >= 0 { 1 } else { -1 }).collect();
    let mut b_work: Vec<i64> = b_int.iter().map(|&b| b.abs()).collect();

    // Process from LSB (site R-1) to MSB (site 0)
    let mut carries: Vec<Vec<i64>> = vec![vec![0i64; m]];
    let mut core_data_list: Vec<AffineCoreData> = Vec::with_capacity(r);

    for _site in (0..r).rev() {
        // Extract current bit: (b_work & 1) * bsign
        let b_curr: Vec<i64> = b_work
            .iter()
            .zip(bsign.iter())
            .map(|(&b, &s)| (b & 1) * s)
            .collect();

        let core_data = affine_transform_core(a_int, &b_curr, scale, m, n, &carries, true)?;
        carries = core_data.carries_out.clone();
        core_data_list.push(core_data);

        // Shift right
        b_work.iter_mut().for_each(|b| *b >>= 1);
    }

    // core_data_list is now in order: [site R-1, site R-2, ..., site 0]

    // Extension loop: handle remaining bits of b for Open BC
    // When abs(b) >= 2^R, high bits of b contribute to carries that affect validity.
    // Extension tensors have site_dim=1 (activebit=false: only x=0, y=0).
    // We fold them into the MSB tensor as a "cap matrix" (Julia approach).
    let cap_matrix: Option<Vec<f64>> = if affine_needs_extension(bc, &b_work) {
        let mut ext_data_list: Vec<AffineCoreData> = Vec::new();
        while b_work.iter().any(|&b| b > 0) {
            let b_curr: Vec<i64> = b_work
                .iter()
                .zip(bsign.iter())
                .map(|(&b, &s)| (b & 1) * s)
                .collect();

            let core_data = affine_transform_core(a_int, &b_curr, scale, m, n, &carries, false)?;
            carries = core_data.carries_out.clone();
            ext_data_list.push(core_data);

            b_work.iter_mut().for_each(|b| *b >>= 1);
        }

        // Build cap matrix by contracting extension tensors with BC weights.
        // Extension tensors have site_dim=1, so they are carry transition matrices:
        //   ext_matrix[cout_idx, cin_idx] = core_data.tensor[[cout_idx, cin_idx, 0]]
        //
        // Process: outermost (last computed) gets BC weights applied,
        // then multiply inward toward the main tensor chain.

        // Start with BC weights on the final carries
        let bc_weights: Vec<f64> = carries
            .iter()
            .map(|c| affine_boundary_weight(c, bc))
            .collect();

        // Contract extension tensors from outermost to innermost
        // ext_data_list is [innermost, ..., outermost] (order of computation)
        // We process from outermost to innermost
        let mut current_weights = bc_weights;
        for ext_data in ext_data_list.iter().rev() {
            let num_cin = ext_data.tensor.dims()[1];
            let mut new_weights = vec![0.0; num_cin];
            for (cin_idx, nw) in new_weights.iter_mut().enumerate() {
                for (cout_idx, &w) in current_weights.iter().enumerate() {
                    if w != 0.0 && ext_data.tensor.get([cout_idx, cin_idx, 0]) {
                        *nw += w;
                    }
                }
            }
            current_weights = new_weights;
        }

        // current_weights now maps: MSB carry_out index -> effective BC weight
        Some(current_weights)
    } else {
        None
    };

    // Build tensors in the same order, then reverse to get [site 0, site 1, ..., site R-1]
    let mut tensors = Vec::with_capacity(r);

    // Helper: compute BC weight for a carry-out index
    let compute_bc_weight = |cout_idx: usize, core_data: &AffineCoreData| -> Complex64 {
        if let Some(ref cap) = cap_matrix {
            // Extension loop was used: weight comes from cap matrix
            Complex64::new(cap[cout_idx], 0.0)
        } else {
            let carry = &core_data.carries_out[cout_idx];
            Complex64::new(affine_boundary_weight(carry, bc), 0.0)
        }
    };

    for (idx, core_data) in core_data_list.iter().enumerate() {
        // idx=0 corresponds to site R-1 (LSB), idx=R-1 corresponds to site 0 (MSB)
        let actual_site = r - 1 - idx;
        let num_carry_out = core_data.carries_out.len();
        let num_carry_in = core_data.tensor.dims()[1];

        // Tensor shape follows shift.rs pattern:
        // t[left, site, right] where left=carry_out (going left), right=carry_in (from right)
        //
        // - Site 0 (MSB): left_dim=1 (BC applied), right_dim=num_carry (receives from right)
        // - Site R-1 (LSB): left_dim=num_carry (sends to left), right_dim=1 (initial carry=0)
        // - Middle: left_dim=num_carry, right_dim=num_carry
        let is_msb = actual_site == 0;
        let is_lsb = actual_site == r - 1;

        let left_dim = if is_msb { 1 } else { num_carry_out };
        let right_dim = if is_lsb { 1 } else { num_carry_in };

        let mut t: tensor4all_simplett::Tensor3<Complex64> =
            tensor3_zeros(left_dim, site_dim, right_dim);

        if is_lsb && is_msb {
            // R==1: single site case
            for cout_idx in 0..num_carry_out {
                let bc_weight = compute_bc_weight(cout_idx, core_data);

                for site_idx in 0..site_dim {
                    if core_data.tensor.get([cout_idx, 0, site_idx]) {
                        let old = t.get3(0, site_idx, 0);
                        t.set3(0, site_idx, 0, *old + bc_weight);
                    }
                }
            }
        } else if is_lsb {
            // LSB (site R-1): initial carry_in=0, send carry_out to left
            // Shape (num_carry_out, site_dim, 1)
            // core_data.tensor[carry_out_idx, carry_in_idx, site_idx]
            // Only carry_in_idx=0 matters (initial carry is the first entry: zero vector)
            for cout_idx in 0..num_carry_out {
                for site_idx in 0..site_dim {
                    if core_data.tensor.get([cout_idx, 0, site_idx]) {
                        t.set3(cout_idx, site_idx, 0, Complex64::one());
                    }
                }
            }
        } else if is_msb {
            // MSB (site 0): apply BC on carry_out, receive carry from right
            for cout_idx in 0..num_carry_out {
                let bc_weight = compute_bc_weight(cout_idx, core_data);

                for cin_idx in 0..num_carry_in {
                    for site_idx in 0..site_dim {
                        if core_data.tensor.get([cout_idx, cin_idx, site_idx]) {
                            let old = t.get3(0, site_idx, cin_idx);
                            t.set3(0, site_idx, cin_idx, *old + bc_weight);
                        }
                    }
                }
            }
        } else {
            // Middle tensors: receive carry from right, send carry to left
            // Shape (num_carry_out, site_dim, num_carry_in)
            for cout_idx in 0..num_carry_out {
                for cin_idx in 0..num_carry_in {
                    for site_idx in 0..site_dim {
                        if core_data.tensor.get([cout_idx, cin_idx, site_idx]) {
                            t.set3(cout_idx, site_idx, cin_idx, Complex64::one());
                        }
                    }
                }
            }
        }

        tensors.push(t);
    }

    // tensors is in order [site R-1, ..., site 0], reverse to get [site 0, ..., site R-1]
    tensors.reverse();

    Ok(tensors)
}

/// Core tensor data for affine transformation.
///
/// Shape: (num_carry_out, num_carry_in, site_dim)
/// where site_dim = 2^(M+N)
struct AffineCoreData {
    /// Possible outgoing carry vectors
    carries_out: Vec<Vec<i64>>,
    /// Tensor data: tensor[carry_out_idx, carry_in_idx, site_idx]
    tensor: BoolTensor3,
}

/// Compute a single core tensor for the affine transformation.
///
/// The core tensor encodes: 2 * carry_out = A * x + b_curr - scale * y + carry_in
///
/// Returns AffineCoreData containing:
/// - carries_out: list of possible outgoing carry vectors
/// - tensor: shape (num_carry_out, num_carry_in, site_dim)
fn affine_transform_core(
    a_int: &[i64],
    b_curr: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    carries_in: &[Vec<i64>],
    activebit: bool,
) -> Result<AffineCoreData> {
    let mut carry_out_map: HashMap<Vec<i64>, BoolTensor2> = HashMap::new();
    let x_range = if activebit {
        checked_pow2(n, "input variable count")?
    } else {
        1
    };
    let y_range = if activebit {
        checked_pow2(m, "output variable count")?
    } else {
        1
    };
    let site_dim = x_range.checked_mul(y_range).ok_or_else(|| {
        anyhow::anyhow!(
            "affine site dimension overflows usize for {m} output variables and {n} input variables"
        )
    })?;
    let num_carry_in = carries_in.len();

    // Iterate over all input carries
    for (c_idx, carry_in) in carries_in.iter().enumerate() {
        // Iterate over all possible x values (N bits)
        for x_bits in 0..x_range {
            let x: Vec<i64> = (0..n).map(|j| ((x_bits >> j) & 1) as i64).collect();

            // Compute z = A*x + b + carry_in
            let mut z: Vec<i64> = vec![0; m];
            for i in 0..m {
                z[i] = carry_in[i] + b_curr[i];
                for j in 0..n {
                    z[i] += a_int[i + m * j] * x[j];
                }
            }

            if scale % 2 == 1 {
                // Scale is odd: unique y that satisfies condition
                let y: Vec<i64> = z.iter().map(|&zi| zi & 1).collect();

                // When bits are inactive, y must be zero (Julia PR #45 fix)
                if !activebit && y.iter().any(|&yi| yi != 0) {
                    continue;
                }

                let y_bits: usize = y
                    .iter()
                    .enumerate()
                    .map(|(i, &yi)| (yi as usize) << i)
                    .sum();

                // Compute carry_out = (z - scale * y) / 2
                let carry_out: Vec<i64> = z
                    .iter()
                    .zip(y.iter())
                    .map(|(&zi, &yi)| (zi - scale * yi) >> 1)
                    .collect();

                // Site index: y bits in lower positions, x bits in upper positions
                let site_idx = y_bits | (x_bits << m);

                let entry = carry_out_map
                    .entry(carry_out)
                    .or_insert_with(|| BoolTensor2::from_elem([num_carry_in, site_dim], false));
                entry.set([c_idx, site_idx], true);
            } else {
                // Scale is even: z must be even for valid y
                if z.iter().any(|&zi| zi % 2 != 0) {
                    continue;
                }

                // y can be any value
                for y_bits in 0..y_range {
                    let y: Vec<i64> = (0..m).map(|i| ((y_bits >> i) & 1) as i64).collect();

                    // Compute carry_out = (z - scale * y) / 2
                    let carry_out: Vec<i64> = z
                        .iter()
                        .zip(y.iter())
                        .map(|(&zi, &yi)| (zi - scale * yi) >> 1)
                        .collect();

                    let site_idx = y_bits | (x_bits << m);

                    let entry = carry_out_map
                        .entry(carry_out)
                        .or_insert_with(|| BoolTensor2::from_elem([num_carry_in, site_dim], false));
                    entry.set([c_idx, site_idx], true);
                }
            }
        }
    }

    // Convert to sorted vectors for deterministic ordering
    let mut carries_out: Vec<Vec<i64>> = carry_out_map.keys().cloned().collect();
    carries_out.sort();

    let num_carry_out = carries_out.len();

    // Build 3D tensor: (num_carry_out, num_carry_in, site_dim)
    let mut tensor = BoolTensor3::from_elem([num_carry_out, num_carry_in, site_dim], false);
    for (cout_idx, carry) in carries_out.iter().enumerate() {
        let data_2d = &carry_out_map[carry];
        for cin_idx in 0..num_carry_in {
            for site_idx in 0..site_dim {
                tensor.set(
                    [cout_idx, cin_idx, site_idx],
                    data_2d.get([cin_idx, site_idx]),
                );
            }
        }
    }

    Ok(AffineCoreData {
        carries_out,
        tensor,
    })
}

#[cfg(test)]
mod tests;
