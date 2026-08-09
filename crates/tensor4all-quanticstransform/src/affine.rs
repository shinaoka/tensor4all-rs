//! Affine transformation operator: y = A*x + b
//!
//! This implements general affine transformations with rational coefficients.
//! The transformation computes y = A*x + b where A is an M×N rational matrix
//! and b is an M-dimensional rational vector.
//!
//! Based on the algorithm from Quantics.jl/src/affine.jl

use std::collections::HashMap;

use anyhow::Result;
use num_bigint::BigInt;
use num_complex::Complex64;
use num_integer::Integer;
use num_rational::Rational64;
use num_traits::{One, Signed, ToPrimitive, Zero};
use sprs::CsMat;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::LinearizationOrder;
use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops, TensorTrain};

use crate::common::{
    checked_allocation_len, checked_pow2, tensortrain_to_linear_operator_asymmetric,
    try_vec_with_capacity, BoundaryCondition, QuanticsOperator,
};
use tensor4all_simplett::{tensor::Tensor3 as GenericTensor3, tensor3_from_data};

#[derive(Clone, Debug)]
struct BoolTensor<const N: usize> {
    data: Vec<u8>,
    dims: [usize; N],
}

type BoolTensor2 = BoolTensor<2>;
type BoolTensor3 = BoolTensor<3>;

impl<const N: usize> BoolTensor<N> {
    fn from_elem(dims: [usize; N], value: bool) -> Result<Self> {
        let total = checked_allocation_len::<u8>(&dims, "affine boolean tensor")?;
        let mut data = try_vec_with_capacity::<u8>("affine boolean tensor", total)?;
        data.resize(total, u8::from(value));
        Ok(Self { data, dims })
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

fn validate_rational_slice(values: &[Rational64], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| *value.denom() == 0) {
        anyhow::bail!("{name}[{index}] has zero denominator");
    }
    Ok(())
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
            .fold(0u64, |factor, value| factor.gcd(&value.unsigned_abs()));

        if common_factor > 1 {
            let mut coefficients = coefficients;
            for coefficient in &mut coefficients {
                *coefficient = divide_i64_by_u64(*coefficient, common_factor);
            }
            Self {
                coefficients,
                rhs: divide_i64_by_u64(rhs, common_factor),
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
    /// # Errors
    /// Returns an error if any coefficient or the right-hand side has a zero
    /// denominator, if the primitive integer representation cannot fit in
    /// `i64`, or if a backing allocation fails.
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
    /// ).unwrap();
    /// assert_eq!(row.coefficients, vec![1, 2]);
    /// assert_eq!(row.rhs, 3);
    /// ```
    pub fn from_rationals(coefficients: Vec<Rational64>, rhs: Rational64) -> Result<Self> {
        validate_rational_slice(&coefficients, "constraint coefficients")?;
        if *rhs.denom() == 0 {
            anyhow::bail!("constraint rhs has zero denominator");
        }

        let mut denominator_lcm = BigInt::one();
        for coefficient in &coefficients {
            denominator_lcm = denominator_lcm.lcm(&BigInt::from(*coefficient.denom()));
        }
        denominator_lcm = denominator_lcm.lcm(&BigInt::from(*rhs.denom()));

        let mut integer_coefficients =
            try_vec_with_capacity::<BigInt>("constraint coefficient list", coefficients.len())?;
        for coefficient in &coefficients {
            let numerator = BigInt::from(*coefficient.numer());
            let denominator = BigInt::from(*coefficient.denom());
            integer_coefficients.push(numerator * (&denominator_lcm / denominator));
        }
        let integer_rhs =
            BigInt::from(*rhs.numer()) * (&denominator_lcm / BigInt::from(*rhs.denom()));

        normalize_bigint_row(integer_coefficients, integer_rhs)
    }
}

fn divide_i64_by_u64(value: i64, divisor: u64) -> i64 {
    if divisor == (1u64 << 63) {
        if value == i64::MIN {
            -1
        } else {
            0
        }
    } else {
        value / divisor as i64
    }
}

fn normalize_bigint_row(coefficients: Vec<BigInt>, rhs: BigInt) -> Result<LinearConstraintRow> {
    let common_factor = coefficients
        .iter()
        .chain(std::iter::once(&rhs))
        .fold(BigInt::zero(), |factor, value| factor.gcd(value))
        .abs();
    let common_factor = if common_factor > BigInt::one() {
        common_factor
    } else {
        BigInt::one()
    };

    let mut normalized_coefficients =
        try_vec_with_capacity::<i64>("normalized constraint coefficient list", coefficients.len())?;
    for coefficient in coefficients {
        normalized_coefficients.push(
            (coefficient / &common_factor)
                .to_i64()
                .ok_or_else(|| anyhow::anyhow!("normalized constraint coefficient exceeds i64"))?,
        );
    }
    let rhs = (rhs / common_factor)
        .to_i64()
        .ok_or_else(|| anyhow::anyhow!("normalized constraint right-hand side exceeds i64"))?;

    Ok(LinearConstraintRow {
        coefficients: normalized_coefficients,
        rhs,
    })
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
/// assert_eq!(params.m(), 1);
/// assert_eq!(params.n(), 1);
///
/// // 2D rotation: y = [[1,1],[1,-1]] * x
/// // Column-major: [A[0,0], A[1,0], A[0,1], A[1,1]]
/// let params = AffineParams::from_integers(
///     vec![1, 1, 1, -1], vec![0, 0], 2, 2
/// ).unwrap();
/// assert_eq!(params.m(), 2);
/// assert_eq!(params.n(), 2);
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
    a: Vec<Rational64>,
    b: Vec<Rational64>,
    m: usize,
    n: usize,
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
    /// Returns an error when any matrix or translation entry has a zero
    /// denominator, when the matrix/vector lengths do not match the
    /// dimensions, a dimension product overflows, or a power-of-two site
    /// dimension cannot be represented safely.
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
        validate_rational_slice(&a, "affine matrix")?;
        validate_rational_slice(&b, "affine translation")?;

        let expected_a_len = m
            .checked_mul(n)
            .ok_or_else(|| anyhow::anyhow!("Affine matrix dimensions overflow usize: {m} × {n}"))?;
        checked_affine_site_dims(m, n)?;
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
        let params = Self { a, b, m, n };
        params.validate()?;
        Ok(params)
    }

    /// Return the matrix entries in column-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::AffineParams;
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(params.a(), &[1.into()]);
    /// ```
    pub fn a(&self) -> &[Rational64] {
        &self.a
    }

    /// Return the translation vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::AffineParams;
    /// let params = AffineParams::from_integers(vec![1], vec![3], 1, 1).unwrap();
    /// assert_eq!(params.b(), &[3.into()]);
    /// ```
    pub fn b(&self) -> &[Rational64] {
        &self.b
    }

    /// Return the number of output variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::AffineParams;
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(params.m(), 1);
    /// ```
    pub fn m(&self) -> usize {
        self.m
    }

    /// Return the number of input variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::AffineParams;
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(params.n(), 1);
    /// ```
    pub fn n(&self) -> usize {
        self.n
    }

    fn validate(&self) -> Result<()> {
        validate_rational_slice(&self.a, "affine matrix")?;
        validate_rational_slice(&self.b, "affine translation")?;

        let expected_a_len = self.m.checked_mul(self.n).ok_or_else(|| {
            anyhow::anyhow!(
                "Affine matrix dimensions overflow usize: {} × {}",
                self.m,
                self.n
            )
        })?;
        if self.a.len() != expected_a_len {
            return Err(anyhow::anyhow!(
                "Matrix A has {} elements but expected {}×{}={}",
                self.a.len(),
                self.m,
                self.n,
                expected_a_len
            ));
        }
        if self.b.len() != self.m {
            return Err(anyhow::anyhow!(
                "Vector b has {} elements but expected {}",
                self.b.len(),
                self.m
            ));
        }
        checked_affine_site_dims(self.m, self.n)?;
        Ok(())
    }

    /// Create affine parameters from integer matrix and vector.
    ///
    /// Convenience method that converts integer values to rationals.
    ///
    /// # Errors
    /// Returns an error when a backing allocation fails, when the matrix/vector
    /// lengths do not match the dimensions, a dimension product overflows, or
    /// a power-of-two site dimension cannot be represented safely.
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
    /// assert_eq!(params.m(), 2);
    /// assert_eq!(params.n(), 2);
    /// ```
    pub fn from_integers(a: Vec<i64>, b: Vec<i64>, m: usize, n: usize) -> Result<Self> {
        let mut a_rat = try_vec_with_capacity::<Rational64>("affine coefficient list", a.len())?;
        for value in a {
            a_rat.push(Rational64::from_integer(value));
        }
        let mut b_rat = try_vec_with_capacity::<Rational64>("affine translation list", b.len())?;
        for value in b {
            b_rat.push(Rational64::from_integer(value));
        }
        Self::new(a_rat, b_rat, m, n)
    }

    /// Get element A[i, j] (0-indexed).
    #[allow(dead_code)]
    fn get_a(&self, i: usize, j: usize) -> Rational64 {
        self.a[i + self.m * j]
    }

    /// Convert to an exact integer representation by scaling with the LCM of
    /// all denominators.
    fn to_integer_scaled(&self) -> Result<(Vec<BigInt>, Vec<BigInt>, BigInt)> {
        let mut denom_lcm = BigInt::one();
        for r in &self.a {
            denom_lcm = denom_lcm.lcm(&BigInt::from(*r.denom()));
        }
        for r in &self.b {
            denom_lcm = denom_lcm.lcm(&BigInt::from(*r.denom()));
        }

        let scale = |r: &Rational64| {
            let numerator = BigInt::from(*r.numer());
            let denominator = BigInt::from(*r.denom());
            numerator * (&denom_lcm / denominator)
        };
        let mut a_int =
            try_vec_with_capacity::<BigInt>("scaled affine coefficient list", self.a.len())?;
        for value in &self.a {
            a_int.push(scale(value));
        }
        let mut b_int =
            try_vec_with_capacity::<BigInt>("scaled affine translation list", self.b.len())?;
        for value in &self.b {
            b_int.push(scale(value));
        }
        Ok((a_int, b_int, denom_lcm))
    }
}

fn try_reserve_for_push<T>(values: &mut Vec<T>, name: &str) -> Result<()> {
    values
        .try_reserve(1)
        .map_err(|err| anyhow::anyhow!("{name} allocation failed while growing: {err}"))
}

fn affine_boundary_weight(carry: &[BigInt], bc: &[BoundaryCondition]) -> f64 {
    carry
        .iter()
        .zip(bc.iter())
        .map(|(c, &boundary)| match boundary {
            BoundaryCondition::Periodic => 1.0,
            BoundaryCondition::AntiPeriodic => {
                if c.is_even() {
                    1.0
                } else {
                    -1.0
                }
            }
            BoundaryCondition::Open => {
                if c.is_zero() {
                    1.0
                } else {
                    0.0
                }
            }
        })
        .product()
}

fn affine_needs_extension(bc: &[BoundaryCondition], b_work: &[BigInt]) -> bool {
    b_work.iter().any(|b| b > &BigInt::zero())
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
    checked_allocation_len::<Complex64>(&[site_dim], "affine site tensor")?;
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
    checked_allocation_len::<usize>(&[site_dim], "affine permutation")?;
    let mut perm = try_vec_with_capacity::<usize>("affine permutation", site_dim)?;
    for old_idx in 0..site_dim {
        let y_bits = old_idx & (output_dim - 1);
        let x_bits = old_idx >> m;
        perm.push(y_bits * input_dim + x_bits);
    }

    let r = mpo.len();
    let mut new_tensors =
        try_vec_with_capacity::<GenericTensor3<Complex64>>("remapped affine tensor list", r)?;

    for i in 0..r {
        let tensor = mpo.site_tensor(i);
        let left_dim = tensor.left_dim();
        let right_dim = tensor.right_dim();

        let total_size = checked_allocation_len::<Complex64>(
            &[left_dim, site_dim, right_dim],
            "remapped affine tensor",
        )?;
        let mut data = try_vec_with_capacity::<Complex64>("remapped affine tensor", total_size)?;
        data.resize(total_size, Complex64::new(0.0, 0.0));
        let mut t = tensor3_from_data(data, left_dim, site_dim, right_dim)
            .map_err(|err| anyhow::anyhow!("Failed to allocate remapped affine tensor: {err}"))?;
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
/// Returns an error if `params` is invalid, `r == 0`, if `bc.len() != params.m()`,
/// or if a required dimension/allocation exceeds the supported `usize` and
/// `isize::MAX` bounds.
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
    params.validate()?;
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

    let mut input_dims = try_vec_with_capacity::<usize>("affine input site dimensions", r)?;
    input_dims.resize(r, input_dim);
    let mut output_dims = try_vec_with_capacity::<usize>("affine output site dimensions", r)?;
    output_dims.resize(r, output_dim);
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
/// A [`LinearOperator`] whose node `i` has `params.n()` input mappings and
/// `params.m()` output mappings, all with binary dimension.
///
/// # Errors
///
/// Returns an error when `params` is invalid, `r == 0`, when
/// `bc.len() != params.m()`, when a required allocation exceeds its checked
/// byte bound, or when the affine tensor network cannot be constructed.
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

    let mut fused_output_indices =
        try_vec_with_capacity::<Index<DynId, TagSet>>("affine fused output indices", r)?;
    for site in 0..r {
        let mapping = op
            .get_output_mapping(&site)
            .ok_or_else(|| anyhow::anyhow!("missing affine output mapping for site {site}"))?;
        fused_output_indices.push(mapping.true_index.clone());
    }
    let mut fused_input_indices =
        try_vec_with_capacity::<Index<DynId, TagSet>>("affine fused input indices", r)?;
    for site in 0..r {
        let mapping = op
            .get_input_mapping(&site)
            .ok_or_else(|| anyhow::anyhow!("missing affine input mapping for site {site}"))?;
        fused_input_indices.push(mapping.true_index.clone());
    }

    for site in 0..r {
        let mut output_indices =
            try_vec_with_capacity::<Index<DynId, TagSet>>("affine output indices", params.m)?;
        output_indices.extend((0..params.m).map(|_| Index::<DynId, TagSet>::new_dyn(2)));
        op = op.unfuse_output_index(
            &fused_output_indices[site],
            &output_indices,
            LinearizationOrder::ColumnMajor,
        )?;

        let mut input_indices =
            try_vec_with_capacity::<Index<DynId, TagSet>>("affine input indices", params.n)?;
        input_indices.extend((0..params.n).map(|_| Index::<DynId, TagSet>::new_dyn(2)));
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
/// Returns an error when `params` is invalid, `r == 0`, when
/// `bc.len() != params.m()`, or when dense dimensions/allocation sizes cannot
/// be represented safely.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{
///     affine_transform_matrix, AffineParams, BoundaryCondition,
/// };
///
/// let params = AffineParams::from_integers(vec![1], vec![1], 1, 1).unwrap();
/// let matrix = affine_transform_matrix(2, &params, &[BoundaryCondition::Periodic]).unwrap();
/// assert_eq!(matrix.rows(), 4);
/// assert_eq!(matrix.cols(), 4);
/// assert_eq!(matrix.nnz(), 4);
/// for x in 0..4 {
///     assert_eq!(matrix.get((x + 1) % 4, x), Some(&1.0));
/// }
/// ```
pub fn affine_transform_matrix(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<CsMat<f64>> {
    params.validate()?;
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

    let (a_int, b_int, scale) = params.to_integer_scaled()?;
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
    let bit_mask = checked_pow2(r, "number of bits")? - 1;
    let modulus = BigInt::one() << r;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    // Iterate over all (x, y) pairs, matching Julia's approach.
    // For periodic BC with scale > 1, multiple y values can satisfy
    // scale * y ≡ A*x + b (mod 2^R), so we must check all pairs.
    for x_flat in 0..input_size {
        // Decode x_flat to N-dimensional x vector
        // x_flat = x[0] + x[1]*2^R + x[2]*2^(2R) + ...
        let mut x = try_vec_with_capacity::<BigInt>("affine input values", n)?;
        for var in 0..n {
            let bit_shift = var
                .checked_mul(r)
                .ok_or_else(|| anyhow::anyhow!("input bit offset overflows usize"))?;
            x.push(BigInt::from((x_flat >> bit_shift) & bit_mask));
        }

        // Compute v = A*x + b (unscaled) exactly.
        let mut v = try_vec_with_capacity::<BigInt>("affine output values", m)?;
        v.resize(m, BigInt::zero());
        for i in 0..m {
            v[i] = b_int[i].clone();
            for j in 0..n {
                v[i] += &a_int[i + m * j] * &x[j];
            }
        }

        for y_flat in 0..output_size {
            // Decode y_flat to M-dimensional y vector
            let mut y = try_vec_with_capacity::<BigInt>("affine output coordinates", m)?;
            for var in 0..m {
                let bit_shift = var
                    .checked_mul(r)
                    .ok_or_else(|| anyhow::anyhow!("output bit offset overflows usize"))?;
                y.push(BigInt::from((y_flat >> bit_shift) & bit_mask));
            }

            let mut carry = try_vec_with_capacity::<BigInt>("affine carry values", m)?;
            carry.resize(m, BigInt::zero());
            let mut valid = true;
            for i in 0..m {
                let diff = &v[i] - &scale * &y[i];
                match bc[i] {
                    BoundaryCondition::Periodic | BoundaryCondition::AntiPeriodic => {
                        if (&diff % &modulus).is_zero() {
                            carry[i] = &diff / &modulus;
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    BoundaryCondition::Open => {
                        if diff.is_zero() {
                            carry[i] = BigInt::zero();
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
                try_reserve_for_push(&mut rows, "affine matrix row list")?;
                try_reserve_for_push(&mut cols, "affine matrix column list")?;
                try_reserve_for_push(&mut vals, "affine matrix value list")?;
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
    let (a_int, b_int, scale) = params.to_integer_scaled()?;
    let m = params.m;
    let n = params.n;

    // Compute core tensors
    let tensors = affine_transform_tensors(r, &a_int, &b_int, &scale, m, n, bc)?;

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
/// Returns an error when `params` is invalid, `r == 0`, when
/// `bc.len() != params.m()`, when a tensor allocation exceeds the checked
/// byte bound, or when affine tensor construction fails.
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
    params.validate()?;
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

    let (a_int, b_int, scale) = params.to_integer_scaled()?;
    let m = params.m;
    let n = params.n;
    let (_, _, site_dim) = checked_affine_site_dims(m, n)?;

    // Compute fused tensors first
    let fused_tensors = affine_transform_tensors(r, &a_int, &b_int, &scale, m, n, bc)?;

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

    let mut unfused_tensors =
        try_vec_with_capacity::<GenericTensor3<Complex64>>("unfused affine tensor list", r)?;

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

        let total_size = checked_allocation_len::<Complex64>(
            &[left_dim, site_dim, right_dim],
            "unfused affine tensor",
        )?;
        let mut unfused_data =
            try_vec_with_capacity::<Complex64>("unfused affine tensor", total_size)?;
        unfused_data.resize(total_size, Complex64::new(0.0, 0.0));

        for l in 0..left_dim {
            for fused_idx in 0..site_dim {
                for rr in 0..right_dim {
                    let val = tensor.get3(l, fused_idx, rr);
                    if val.norm() > 0.0 {
                        // Tensor3 storage is column-major: axis 0 is minor.
                        let flat_idx = l
                            .checked_add(
                                left_dim
                                    .checked_mul(
                                        fused_idx
                                            .checked_add(
                                                site_dim.checked_mul(rr).ok_or_else(|| {
                                                    anyhow::anyhow!(
                                                        "unfused affine tensor offset overflows usize"
                                                    )
                                                })?,
                                            )
                                            .ok_or_else(|| {
                                                anyhow::anyhow!(
                                                    "unfused affine tensor offset overflows usize"
                                                )
                                            })?,
                                    )
                                    .ok_or_else(|| {
                                        anyhow::anyhow!(
                                            "unfused affine tensor offset overflows usize"
                                        )
                                    })?,
                            )
                            .ok_or_else(|| {
                                anyhow::anyhow!("unfused affine tensor offset overflows usize")
                            })?;
                        unfused_data[flat_idx] = *val;
                    }
                }
            }
        }

        let unfused_tensor = tensor3_from_data(unfused_data, left_dim, site_dim, right_dim)
            .map_err(|err| anyhow::anyhow!("Failed to allocate unfused affine tensor: {err}"))?;
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
/// let info = UnfusedTensorInfo::new(&params).unwrap();
///
/// assert_eq!(info.m(), 2);
/// assert_eq!(info.n(), 2);
/// assert_eq!(info.num_physical_dims(), 4);
///
/// // Get shape for a tensor with bond dims 3 and 5
/// let shape = info.unfused_shape(3, 5).unwrap();
/// assert_eq!(shape, vec![3, 2, 2, 2, 2, 5]);
///
/// // Round-trip encode/decode
/// let fused = info.encode_fused_index(&[1, 0], &[0, 1]).unwrap();
/// let (y_bits, x_bits) = info.decode_fused_index(fused).unwrap();
/// assert_eq!(y_bits, vec![1, 0]);
/// assert_eq!(x_bits, vec![0, 1]);
/// ```
#[derive(Clone, Debug)]
pub struct UnfusedTensorInfo {
    m: usize,
    n: usize,
    num_physical_dims: usize,
    physical_dim: usize,
    site_dim: usize,
}

impl UnfusedTensorInfo {
    /// Create checked metadata for the given affine parameters.
    ///
    /// # Errors
    /// Returns an error if the affine parameters are malformed, a variable
    /// width cannot be represented by `usize`, the fused site dimension
    /// overflows, or `M + N` cannot be represented safely.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// let info = UnfusedTensorInfo::new(&params).unwrap();
    /// assert_eq!(info.site_dim(), 4);
    /// ```
    pub fn new(params: &AffineParams) -> Result<Self> {
        let num_physical_dims = params
            .m
            .checked_add(params.n)
            .ok_or_else(|| anyhow::anyhow!("unfused physical dimension count overflows usize"))?;
        params.validate()?;
        let (_, _, site_dim) = checked_affine_site_dims(params.m, params.n)?;
        num_physical_dims
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("unfused shape rank overflows usize"))?;
        Ok(Self {
            m: params.m,
            n: params.n,
            num_physical_dims,
            physical_dim: 2,
            site_dim,
        })
    }

    /// Return the number of output variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(UnfusedTensorInfo::new(&params).unwrap().m(), 1);
    /// ```
    pub fn m(&self) -> usize {
        self.m
    }

    /// Return the number of input variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(UnfusedTensorInfo::new(&params).unwrap().n(), 1);
    /// ```
    pub fn n(&self) -> usize {
        self.n
    }

    /// Return the total number of binary physical dimensions per site.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(UnfusedTensorInfo::new(&params).unwrap().num_physical_dims(), 2);
    /// ```
    pub fn num_physical_dims(&self) -> usize {
        self.num_physical_dims
    }

    /// Return the dimension of each unfused physical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(UnfusedTensorInfo::new(&params).unwrap().physical_dim(), 2);
    /// ```
    pub fn physical_dim(&self) -> usize {
        self.physical_dim
    }

    /// Return the fused site dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// assert_eq!(UnfusedTensorInfo::new(&params).unwrap().site_dim(), 4);
    /// ```
    pub fn site_dim(&self) -> usize {
        self.site_dim
    }

    /// Get the shape for a fully unfused tensor at a given site.
    ///
    /// Returns `[left_bond, 2, 2, ..., 2, right_bond]` where there are M+N 2s.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// let info = UnfusedTensorInfo::new(&params).unwrap();
    /// assert_eq!(info.unfused_shape(2, 3).unwrap(), vec![2, 2, 2, 3]);
    /// ```
    ///
    /// # Errors
    /// Returns an error if the shape rank or backing allocation cannot be
    /// represented safely.
    pub fn unfused_shape(&self, left_bond: usize, right_bond: usize) -> Result<Vec<usize>> {
        let rank = self
            .num_physical_dims
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("unfused shape rank overflows usize"))?;
        let mut shape = try_vec_with_capacity::<usize>("unfused shape", rank)?;
        shape.push(left_bond);
        shape.extend(std::iter::repeat_n(2, self.num_physical_dims));
        shape.push(right_bond);
        Ok(shape)
    }

    /// Decode a fused site index to individual variable bits.
    ///
    /// # Errors
    /// Returns an error when `fused_idx` is outside the checked fused site
    /// dimension or when a decoded-bit backing allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// let info = UnfusedTensorInfo::new(&params).unwrap();
    /// assert_eq!(info.decode_fused_index(3).unwrap(), (vec![1], vec![1]));
    /// ```
    pub fn decode_fused_index(&self, fused_idx: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        if fused_idx >= self.site_dim {
            return Err(anyhow::anyhow!(
                "fused index {fused_idx} is outside site dimension {}",
                self.site_dim
            ));
        }
        let output_dim = checked_pow2(self.m, "output variable count")?;
        let y_combined = fused_idx & (output_dim - 1);
        let x_combined = fused_idx >> self.m;

        let mut y_bits = try_vec_with_capacity::<usize>("decoded affine output bits", self.m)?;
        for i in 0..self.m {
            y_bits.push((y_combined >> i) & 1);
        }
        let mut x_bits = try_vec_with_capacity::<usize>("decoded affine input bits", self.n)?;
        for j in 0..self.n {
            x_bits.push((x_combined >> j) & 1);
        }

        Ok((y_bits, x_bits))
    }

    /// Encode individual variable bits to a fused site index.
    ///
    /// # Arguments
    /// * `y_bits` - Bits for output variables (length M), each either 0 or 1.
    /// * `x_bits` - Bits for input variables (length N), each either 0 or 1.
    ///
    /// # Errors
    /// Returns an error when either slice has the wrong length, contains a
    /// value other than 0 or 1, or the encoded index cannot be represented.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstransform::{AffineParams, UnfusedTensorInfo};
    /// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    /// let info = UnfusedTensorInfo::new(&params).unwrap();
    /// assert_eq!(info.encode_fused_index(&[1], &[1]).unwrap(), 3);
    /// assert!(info.encode_fused_index(&[2], &[0]).is_err());
    /// ```
    pub fn encode_fused_index(&self, y_bits: &[usize], x_bits: &[usize]) -> Result<usize> {
        if y_bits.len() != self.m || x_bits.len() != self.n {
            return Err(anyhow::anyhow!(
                "fused index bit lengths must be {} output and {} input bits",
                self.m,
                self.n
            ));
        }

        let mut y_combined = 0usize;
        for (i, &bit) in y_bits.iter().enumerate() {
            if bit > 1 {
                return Err(anyhow::anyhow!("output bit {i} must be 0 or 1, got {bit}"));
            }
            let shift = u32::try_from(i)
                .map_err(|_| anyhow::anyhow!("output bit shift {i} exceeds u32"))?;
            y_combined |= bit
                .checked_shl(shift)
                .ok_or_else(|| anyhow::anyhow!("output bit shift {i} overflows usize"))?;
        }

        let mut x_combined = 0usize;
        for (j, &bit) in x_bits.iter().enumerate() {
            if bit > 1 {
                return Err(anyhow::anyhow!("input bit {j} must be 0 or 1, got {bit}"));
            }
            let shift =
                u32::try_from(j).map_err(|_| anyhow::anyhow!("input bit shift {j} exceeds u32"))?;
            x_combined |= bit
                .checked_shl(shift)
                .ok_or_else(|| anyhow::anyhow!("input bit shift {j} overflows usize"))?;
        }

        let shift = u32::try_from(self.m)
            .map_err(|_| anyhow::anyhow!("output variable count exceeds u32 shift width"))?;
        let encoded = x_combined
            .checked_shl(shift)
            .map(|x| x | y_combined)
            .ok_or_else(|| anyhow::anyhow!("fused index overflows usize"))?;
        if encoded >= self.site_dim {
            return Err(anyhow::anyhow!(
                "encoded fused index {encoded} is outside site dimension {}",
                self.site_dim
            ));
        }
        Ok(encoded)
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
    a_int: &[BigInt],
    b_int: &[BigInt],
    scale: &BigInt,
    m: usize,
    n: usize,
    bc: &[BoundaryCondition],
) -> Result<Vec<tensor4all_simplett::Tensor3<Complex64>>> {
    let (_, _, site_dim) = checked_affine_site_dims(m, n)?;

    // Track sign separately and work with absolute value so that right-shifting
    // always terminates. BigInt keeps all coefficient arithmetic exact.
    let mut bsign = try_vec_with_capacity::<i8>("affine translation signs", m)?;
    let mut b_work = try_vec_with_capacity::<BigInt>("affine translation magnitudes", m)?;
    for b in b_int {
        bsign.push(if b.is_negative() { -1 } else { 1 });
        b_work.push(b.abs());
    }

    // Process from LSB (site R-1) to MSB (site 0)
    let mut initial_carry = try_vec_with_capacity::<BigInt>("affine initial carry", m)?;
    initial_carry.resize(m, BigInt::zero());
    let mut core_data_list = try_vec_with_capacity::<AffineCoreData>("affine core list", r)?;

    for _site in (0..r).rev() {
        // Extract current bit: (b_work & 1) * bsign
        let mut b_curr = try_vec_with_capacity::<BigInt>("affine current translation bits", m)?;
        for (b, &s) in b_work.iter().zip(bsign.iter()) {
            b_curr.push(if (b % 2u8).is_zero() {
                BigInt::zero()
            } else {
                BigInt::from(s)
            });
        }

        // Reborrow the previous core's carry vectors instead of cloning every
        // BigInt-backed carry list between neighboring sites.
        let core_data = if let Some(previous) = core_data_list.last() {
            affine_transform_core(a_int, &b_curr, scale, m, n, &previous.carries_out, true)?
        } else {
            affine_transform_core(
                a_int,
                &b_curr,
                scale,
                m,
                n,
                std::slice::from_ref(&initial_carry),
                true,
            )?
        };
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
        while b_work.iter().any(|b| b > &BigInt::zero()) {
            let mut b_curr =
                try_vec_with_capacity::<BigInt>("affine extension translation bits", m)?;
            for (b, &s) in b_work.iter().zip(bsign.iter()) {
                b_curr.push(if (b % 2u8).is_zero() {
                    BigInt::zero()
                } else {
                    BigInt::from(s)
                });
            }

            // Continue from the last main/extension core without cloning its
            // BigInt-backed carry vectors.
            let core_data = if let Some(previous) = ext_data_list.last() {
                affine_transform_core(a_int, &b_curr, scale, m, n, &previous.carries_out, false)?
            } else {
                let previous = core_data_list
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("affine extension requires a preceding core"))?;
                affine_transform_core(a_int, &b_curr, scale, m, n, &previous.carries_out, false)?
            };
            try_reserve_for_push(&mut ext_data_list, "affine extension core list")?;
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
        let final_carries: &[Vec<BigInt>] = if let Some(core) = ext_data_list.last() {
            core.carries_out.as_slice()
        } else {
            &core_data_list
                .last()
                .ok_or_else(|| anyhow::anyhow!("affine main core list must be non-empty"))?
                .carries_out
        };
        let mut bc_weights =
            try_vec_with_capacity::<f64>("affine boundary weights", final_carries.len())?;
        for carry in final_carries {
            bc_weights.push(affine_boundary_weight(carry, bc));
        }

        // Contract extension tensors from outermost to innermost
        // ext_data_list is [innermost, ..., outermost] (order of computation)
        // We process from outermost to innermost
        let mut current_weights = bc_weights;
        for ext_data in ext_data_list.iter().rev() {
            let num_cin = ext_data.tensor.dims()[1];
            let mut new_weights =
                try_vec_with_capacity::<f64>("affine extension weights", num_cin)?;
            new_weights.resize(num_cin, 0.0);
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
    let mut tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "affine MPO tensor list",
        r,
    )?;

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

        let total_size = checked_allocation_len::<Complex64>(
            &[left_dim, site_dim, right_dim],
            "affine MPO tensor",
        )?;
        let mut data = try_vec_with_capacity::<Complex64>("affine MPO tensor", total_size)?;
        data.resize(total_size, Complex64::new(0.0, 0.0));
        let mut t: tensor4all_simplett::Tensor3<Complex64> =
            tensor3_from_data(data, left_dim, site_dim, right_dim)
                .map_err(|err| anyhow::anyhow!("Failed to allocate affine MPO tensor: {err}"))?;

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
    carries_out: Vec<Vec<BigInt>>,
    /// Tensor data: tensor[carry_out_idx, carry_in_idx, site_idx]
    tensor: BoolTensor3,
}

fn record_affine_core_transition(
    carry_out_map: &mut HashMap<Vec<BigInt>, BoolTensor2>,
    carry_out: &mut Vec<BigInt>,
    carry_len: usize,
    num_carry_in: usize,
    site_dim: usize,
    carry_in_idx: usize,
    site_idx: usize,
) -> Result<()> {
    if let Some(entry) = carry_out_map.get_mut(&*carry_out) {
        entry.set([carry_in_idx, site_idx], true);
        return Ok(());
    }

    let mut data = BoolTensor2::from_elem([num_carry_in, site_dim], false)?;
    data.set([carry_in_idx, site_idx], true);
    carry_out_map.try_reserve(1).map_err(|err| {
        anyhow::anyhow!("affine carry transition map allocation failed while growing: {err}")
    })?;
    carry_out_map.insert(std::mem::take(carry_out), data);
    carry_out.try_reserve_exact(carry_len).map_err(|err| {
        anyhow::anyhow!("affine carry scratch allocation failed while resetting: {err}")
    })?;
    carry_out.resize(carry_len, BigInt::zero());
    Ok(())
}

/// Compute a single core tensor for the affine transformation.
///
/// The core tensor encodes: 2 * carry_out = A * x + b_curr - scale * y + carry_in
///
/// Returns AffineCoreData containing:
/// - carries_out: list of possible outgoing carry vectors
/// - tensor: shape (num_carry_out, num_carry_in, site_dim)
fn affine_transform_core(
    a_int: &[BigInt],
    b_curr: &[BigInt],
    scale: &BigInt,
    m: usize,
    n: usize,
    carries_in: &[Vec<BigInt>],
    activebit: bool,
) -> Result<AffineCoreData> {
    let mut carry_out_map: HashMap<Vec<BigInt>, BoolTensor2> = HashMap::new();
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
    let variable_shift = u32::try_from(m)
        .map_err(|_| anyhow::anyhow!("output variable count exceeds u32 shift width"))?;

    // Iterate over all input carries.
    for (c_idx, carry_in) in carries_in.iter().enumerate() {
        // Reuse these buffers across all x/y states. The map owns a carry vector
        // only when it is a new key; duplicate transitions keep the scratch
        // allocation in place.
        let mut z = try_vec_with_capacity::<BigInt>("affine core scratch", m)?;
        z.resize(m, BigInt::zero());
        let mut carry_out = try_vec_with_capacity::<BigInt>("affine core carry scratch", m)?;
        carry_out.resize(m, BigInt::zero());

        // Iterate over all possible x values (N bits).
        for x_bits in 0..x_range {
            // Compute z = A*x + b + carry_in exactly. Read x bits directly so
            // coefficients are only added for set bits; this avoids constructing
            // a BigInt-valued x vector and multiplying by zero or one.
            for i in 0..m {
                z[i].clone_from(&carry_in[i]);
                z[i] += &b_curr[i];
                for j in 0..n {
                    if ((x_bits >> j) & 1) != 0 {
                        z[i] += &a_int[i + m * j];
                    }
                }
            }

            let shifted_x = x_bits
                .checked_shl(variable_shift)
                .ok_or_else(|| anyhow::anyhow!("affine site index overflows usize"))?;

            if scale.is_odd() {
                // Scale is odd: unique y that satisfies the condition. Build
                // its packed bit index directly from z's parity and reuse the
                // carry-out scratch vector.
                let mut y_bits = 0usize;
                for i in 0..m {
                    carry_out[i].clone_from(&z[i]);
                    if z[i].is_odd() {
                        if !activebit {
                            y_bits = 0;
                            break;
                        }
                        let shift = u32::try_from(i)
                            .map_err(|_| anyhow::anyhow!("output bit shift exceeds u32"))?;
                        y_bits |= 1usize
                            .checked_shl(shift)
                            .ok_or_else(|| anyhow::anyhow!("output bit shift overflows usize"))?;
                        carry_out[i] -= scale;
                    }
                    carry_out[i] >>= 1usize;
                }

                // When bits are inactive, y must be zero (Julia PR #45 fix).
                if !activebit && z.iter().any(|zi| zi.is_odd()) {
                    continue;
                }

                record_affine_core_transition(
                    &mut carry_out_map,
                    &mut carry_out,
                    m,
                    num_carry_in,
                    site_dim,
                    c_idx,
                    y_bits | shifted_x,
                )?;
            } else {
                // Scale is even: z must be even for a valid y.
                if z.iter().any(|zi| zi.is_odd()) {
                    continue;
                }

                // y can be any value. Subtract scale only for set output bits,
                // then shift in place; no per-state BigInt bit vector needed.
                for y_bits in 0..y_range {
                    for i in 0..m {
                        carry_out[i].clone_from(&z[i]);
                        if ((y_bits >> i) & 1) != 0 {
                            carry_out[i] -= scale;
                        }
                        carry_out[i] >>= 1usize;
                    }

                    record_affine_core_transition(
                        &mut carry_out_map,
                        &mut carry_out,
                        m,
                        num_carry_in,
                        site_dim,
                        c_idx,
                        y_bits | shifted_x,
                    )?;
                }
            }
        }
    }

    // Move the map entries into a fallibly reserved outer vector. Sorting the
    // entries keeps deterministic carry ordering without cloning any BigInts.
    let mut carry_entries = try_vec_with_capacity::<(Vec<BigInt>, BoolTensor2)>(
        "affine outgoing carry entries",
        carry_out_map.len(),
    )?;
    for entry in carry_out_map {
        carry_entries.push(entry);
    }
    carry_entries.sort_by(|(left, _), (right, _)| left.cmp(right));

    let num_carry_out = carry_entries.len();

    // Build 3D tensor: (num_carry_out, num_carry_in, site_dim)
    let mut tensor = BoolTensor3::from_elem([num_carry_out, num_carry_in, site_dim], false)?;
    for (cout_idx, (_, data_2d)) in carry_entries.iter().enumerate() {
        for cin_idx in 0..num_carry_in {
            for site_idx in 0..site_dim {
                tensor.set(
                    [cout_idx, cin_idx, site_idx],
                    data_2d.get([cin_idx, site_idx]),
                );
            }
        }
    }

    let mut carries_out =
        try_vec_with_capacity::<Vec<BigInt>>("affine outgoing carry list", num_carry_out)?;
    for (carry, _) in carry_entries {
        carries_out.push(carry);
    }

    Ok(AffineCoreData {
        carries_out,
        tensor,
    })
}

#[cfg(test)]
mod tests;
