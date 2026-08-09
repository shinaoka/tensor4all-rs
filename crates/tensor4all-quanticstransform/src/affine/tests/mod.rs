use super::*;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};

fn tensor_axis_lookup(template: &TensorDynLen) -> std::collections::HashMap<DynIndex, usize> {
    template
        .indices()
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, index)| (index, position))
        .collect()
}

fn tensor_axis_position(
    lookup: &std::collections::HashMap<DynIndex, usize>,
    index: &DynIndex,
) -> usize {
    lookup
        .get(index)
        .copied()
        .expect("tensor axis index not found in contracted tensor")
}

#[test]
fn tensor_axis_lookup_uses_full_index_identity() {
    let base = DynIndex::new_dyn(2);
    let primed = base.prime();
    let tensor = TensorDynLen::from_dense(
        vec![base.clone(), primed.clone()],
        vec![Complex64::new(0.0, 0.0); 4],
    )
    .unwrap();
    let lookup = tensor_axis_lookup(&tensor);

    assert_eq!(lookup.get(&base), Some(&0));
    assert_eq!(lookup.get(&primed), Some(&1));
}

#[test]
fn test_affine_params_new() {
    let a = vec![
        Rational64::from_integer(1),
        Rational64::from_integer(0),
        Rational64::from_integer(0),
        Rational64::from_integer(1),
    ];
    let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
    let params = AffineParams::new(a, b, 2, 2);
    assert!(params.is_ok());
}

#[test]
fn test_linear_constraint_row_accepts_i64_extremes() {
    let row = LinearConstraintRow::from_integers(vec![i64::MIN], i64::MIN);
    assert_eq!(row.coefficients, vec![-1]);
    assert_eq!(row.rhs, -1);

    let error = LinearConstraintRow::from_rationals(
        vec![Rational64::new(i64::MAX, 2)],
        Rational64::new(1, 3),
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds i64"));
}

#[test]
fn test_rational_inputs_reject_zero_denominators() {
    let malformed = Rational64::new_raw(1, 0);

    let error =
        AffineParams::new(vec![malformed], vec![Rational64::from_integer(0)], 1, 1).unwrap_err();
    assert!(error.to_string().contains("zero denominator"));

    let error = LinearConstraintRow::from_rationals(vec![malformed], Rational64::from_integer(1))
        .unwrap_err();
    assert!(error.to_string().contains("zero denominator"));

    let error = LinearConstraintRow::from_rationals(vec![Rational64::from_integer(1)], malformed)
        .unwrap_err();
    assert!(error.to_string().contains("zero denominator"));
}

#[test]
fn test_affine_extreme_i64_coefficients_are_exact() {
    for &(coefficient, translation) in &[
        (i64::MIN, i64::MIN),
        (i64::MIN, i64::MAX),
        (i64::MAX, i64::MIN),
        (i64::MAX, i64::MAX),
    ] {
        let params =
            AffineParams::from_integers(vec![coefficient], vec![translation], 1, 1).unwrap();
        let matrix = affine_transform_matrix(2, &params, &[BoundaryCondition::Periodic]).unwrap();
        let operator = affine_operator(2, &params, &[BoundaryCondition::Periodic]).unwrap();
        let actual = operator.mpo().contract_to_tensor().unwrap();
        let expected = affine_operator_expected_dense_tensor(&operator, &actual, 2, |y, x| {
            let expected_y =
                ((coefficient as i128) * (x as i128) + translation as i128).rem_euclid(4) as usize;
            Complex64::new(f64::from((y == expected_y) as u8), 0.0)
        });
        let maxabs = actual.sub(&expected).unwrap().maxabs().unwrap();
        assert!(
            maxabs < 1e-12,
            "MPO mismatch for a={coefficient}, b={translation}: {maxabs}"
        );

        for x in 0..4usize {
            let expected_y =
                ((coefficient as i128) * (x as i128) + translation as i128).rem_euclid(4) as usize;
            for y in 0..4usize {
                let expected = f64::from((y == expected_y) as u8);
                assert_eq!(
                    *matrix.get(y, x).unwrap_or(&0.0),
                    expected,
                    "reference matrix entry ({y}, {x}) for a={coefficient}, b={translation}"
                );
            }
        }
    }
}

#[test]
fn test_affine_params_from_integers() {
    let a = vec![1, 0, 0, 1];
    let b = vec![0, 0];
    let params = AffineParams::from_integers(a, b, 2, 2);
    assert!(params.is_ok());
}

#[test]
fn test_affine_params_column_major_indexing() {
    // Matrix [[1, 2, 3], [4, 5, 6]] stored column-major.
    let params = AffineParams::from_integers(vec![1, 4, 2, 5, 3, 6], vec![0, 0], 2, 3).unwrap();

    assert_eq!(params.get_a(0, 0), Rational64::from_integer(1));
    assert_eq!(params.get_a(1, 0), Rational64::from_integer(4));
    assert_eq!(params.get_a(0, 1), Rational64::from_integer(2));
    assert_eq!(params.get_a(1, 1), Rational64::from_integer(5));
    assert_eq!(params.get_a(0, 2), Rational64::from_integer(3));
    assert_eq!(params.get_a(1, 2), Rational64::from_integer(6));
}

#[test]
fn test_affine_params_to_integer_scaled() {
    // Test with rational coefficients
    let a = vec![
        Rational64::new(1, 2), // 1/2
        Rational64::new(1, 3), // 1/3
    ];
    let b = vec![Rational64::new(1, 6)]; // 1/6
    let params = AffineParams::new(a, b, 1, 2).unwrap();

    let (a_int, b_int, scale) = params.to_integer_scaled().unwrap();

    // LCM of denominators (2, 3, 6) = 6
    assert_eq!(scale, BigInt::from(6));
    assert_eq!(a_int, vec![BigInt::from(3), BigInt::from(2)]); // [1/2 * 6, 1/3 * 6]
    assert_eq!(b_int, vec![BigInt::from(1)]); // [1/6 * 6]
}

#[test]
fn test_linear_constraint_row_from_integers_divides_common_factor() {
    let row = LinearConstraintRow::from_integers(vec![16], 64);

    assert_eq!(row.coefficients, vec![1]);
    assert_eq!(row.rhs, 4);
}

#[test]
fn test_linear_constraint_row_from_rationals_clears_denominators_then_reduces() {
    let row = LinearConstraintRow::from_rationals(
        vec![Rational64::new(2, 3), Rational64::new(4, 3)],
        Rational64::from_integer(2),
    )
    .unwrap();

    assert_eq!(row.coefficients, vec![1, 2]);
    assert_eq!(row.rhs, 3);
}

#[test]
fn test_linear_constraint_row_uses_positive_gcd_for_negative_rows() {
    let row = LinearConstraintRow::from_integers(vec![-16], -64);

    assert_eq!(row.coefficients, vec![-1]);
    assert_eq!(row.rhs, -4);
}

#[test]
fn test_linear_constraint_row_preserves_all_zero_row() {
    let row = LinearConstraintRow::from_integers(vec![0, 0], 0);

    assert_eq!(row.coefficients, vec![0, 0]);
    assert_eq!(row.rhs, 0);
}

#[test]
fn test_affine_params_to_integer_scaled_does_not_normalize_constraint_rows() {
    let params = AffineParams::from_integers(vec![16], vec![64], 1, 1).unwrap();

    let (a_int, b_int, scale) = params.to_integer_scaled().unwrap();

    assert_eq!(a_int, vec![BigInt::from(16)]);
    assert_eq!(b_int, vec![BigInt::from(64)]);
    assert_eq!(scale, BigInt::from(1));
}

#[test]
fn test_affine_transform_identity() {
    // Identity transformation: y = x
    let a = vec![1i64];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_operator_creation() {
    // Simple 2D transformation
    let a = vec![1i64, 1, 1, -1]; // [[1, 1], [1, -1]] in column-major
    let b = vec![0i64, 0];
    let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_pullback_operator_creation_via_transpose() {
    // Verify we can build the pullback of a 1->2 affine map via
    // affine_operator(...).transpose(): the pullback direction swaps
    // input/output roles without panicking or erroring.
    let a = vec![1i64, 0];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 2).unwrap();
    let bc = vec![BoundaryCondition::Open];

    let op = affine_operator(4, &params, &bc);
    assert!(op.is_ok());
    let _transposed = op.unwrap().transpose();
}

#[test]
fn test_affine_error_zero_bits() {
    let a = vec![1i64];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(0, &params, &bc);
    assert!(result.is_err());
}

#[test]
fn test_affine_error_bc_mismatch() {
    let a = vec![1i64, 0, 0, 1];
    let b = vec![0i64, 0];
    let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic]; // Only 1 BC but M=2

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_err());
}

#[test]
fn test_affine_operator_rejects_variable_shift_width() {
    let params = AffineParams {
        a: vec![Rational64::from_integer(0); usize::BITS as usize],
        b: vec![Rational64::from_integer(0)],
        m: 1,
        n: usize::BITS as usize,
    };
    let error = affine_operator(1, &params, &[BoundaryCondition::Periodic]).unwrap_err();
    assert!(error.to_string().contains("input variable count"));
}

#[test]
fn test_affine_operator_rejects_site_dimension_overflow() {
    let width = usize::BITS as usize / 2;
    let params = AffineParams {
        a: vec![Rational64::from_integer(0); width * width],
        b: vec![Rational64::from_integer(0); width],
        m: width,
        n: width,
    };
    let bc = vec![BoundaryCondition::Periodic; width];
    let error = affine_operator(1, &params, &bc).unwrap_err();
    assert!(error.to_string().contains("site dimension"));
}

#[test]
fn test_affine_params_reject_site_byte_limit_before_tensor_allocation() {
    let width = usize::BITS as usize / 2 - 1;
    let error = AffineParams::new(
        vec![Rational64::from_integer(0); width * width],
        vec![Rational64::from_integer(0); width],
        width,
        width,
    )
    .unwrap_err();
    assert!(error.to_string().contains("byte length"));
}

#[test]
fn test_affine_pullback_bc_mismatch_via_transpose() {
    // Boundary-condition length must equal params.m(); affine_operator
    // rejects mismatches regardless of whether the caller later transposes.
    let a = vec![1i64, 0];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_err());
}

#[test]
fn test_affine_params_dimension_error() {
    // a.len() != m * n
    let a = vec![Rational64::from_integer(1), Rational64::from_integer(0)]; // 2 elements
    let b = vec![Rational64::from_integer(0)];
    let params = AffineParams::new(a, b, 2, 2); // expects 4 elements
    assert!(params.is_err());

    // b.len() != m
    let a = vec![
        Rational64::from_integer(1),
        Rational64::from_integer(0),
        Rational64::from_integer(0),
        Rational64::from_integer(1),
    ];
    let b = vec![Rational64::from_integer(0)]; // 1 element but m=2
    let params = AffineParams::new(a, b, 2, 2);
    assert!(params.is_err());

    let error = AffineParams::new(Vec::new(), Vec::new(), usize::MAX, 2).unwrap_err();
    assert!(error.to_string().contains("overflow"));

    let error = AffineParams::new(
        Vec::new(),
        vec![Rational64::from_integer(0)],
        1,
        usize::BITS as usize,
    )
    .unwrap_err();
    assert!(error.to_string().contains("input variable count"));
}

#[test]
fn test_affine_with_rational_coefficients() {
    // y = (1/2)*x
    let a = vec![Rational64::new(1, 2)];
    let b = vec![Rational64::from_integer(0)];
    let params = AffineParams::new(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_shift_only() {
    // y = x + 3
    let a = vec![1i64];
    let b = vec![3i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_scale_by_two() {
    // y = 2*x
    let a = vec![2i64];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_asymmetric_dimensions() {
    // M=1, N=2: y = x1 + x2 (sum of two inputs to one output)
    let a = vec![1i64, 1]; // 1×2 matrix
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_open_boundary() {
    // Identity with open boundary
    let a = vec![1i64];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Open];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_negation() {
    // y = -x
    let a = vec![-1i64];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_ok());
}

#[test]
fn test_affine_mpo_structure() {
    // Verify MPO tensor structure for identity transform
    let a = vec![1i64];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let op = affine_operator(4, &params, &bc).unwrap();
    // Check that the operator was created successfully
    // (More detailed structure tests would require accessing internal TreeTN)
    let _ = op;
}

#[test]
fn test_affine_larger_bits() {
    // Test with more bits
    let a = vec![1i64];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let result = affine_operator(8, &params, &bc);
    assert!(result.is_ok());

    let result = affine_operator(16, &params, &bc);
    assert!(result.is_ok());
}

// ========== Matrix verification tests ==========

#[test]
fn test_affine_matrix_identity() {
    // Identity transformation: y = x
    let r = 3;
    let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

    // Should be identity matrix of size 2^R × 2^R
    let size = 1 << r;
    assert_eq!(matrix.rows(), size);
    assert_eq!(matrix.cols(), size);
    assert_eq!(matrix.nnz(), size); // Identity has exactly N non-zeros

    // Check that it's identity
    for i in 0..size {
        assert_eq!(*matrix.get(i, i).unwrap_or(&0.0), 1.0);
    }
}

#[test]
fn test_affine_matrix_shift() {
    // Shift transformation: y = x + 3 (mod 2^R)
    let r = 3;
    let params = AffineParams::from_integers(vec![1], vec![3], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

    let size = 1 << r; // 8
    assert_eq!(matrix.nnz(), size); // Permutation has exactly N non-zeros

    // Check specific mappings: y = (x + 3) mod 8
    // x=0 -> y=3, x=1 -> y=4, ..., x=5 -> y=0, x=6 -> y=1, x=7 -> y=2
    for x in 0..size {
        let y = (x + 3) % size;
        assert_eq!(*matrix.get(y, x).unwrap_or(&0.0), 1.0);
    }
}

#[test]
fn test_affine_matrix_scale_by_two() {
    // Scale: y = 2*x (mod 2^R)
    let r = 3;
    let params = AffineParams::from_integers(vec![2], vec![0], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

    let size = 1 << r; // 8
    assert_eq!(matrix.nnz(), size);

    // Check: y = 2*x mod 8
    for x in 0..size {
        let y = (2 * x) % size;
        assert_eq!(*matrix.get(y, x).unwrap_or(&0.0), 1.0);
    }
}

#[test]
fn test_affine_matrix_sum_2d() {
    // y = x1 + x2 (M=1, N=2)
    let r = 2;
    let params = AffineParams::from_integers(vec![1, 1], vec![0], 1, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

    let input_size = 1 << (r * 2); // 2^(2*2) = 16
    let output_size = 1 << r; // 2^2 = 4

    assert_eq!(matrix.rows(), output_size);
    assert_eq!(matrix.cols(), input_size);

    // Check specific cases
    // x_flat = x1 + x2 * 2^R
    // x1=1, x2=2: x_flat = 1 + 2*4 = 9, y = (1+2) mod 4 = 3
    assert_eq!(*matrix.get(3, 9).unwrap_or(&0.0), 1.0);
    // x1=3, x2=3: x_flat = 3 + 3*4 = 15, y = (3+3) mod 4 = 2
    assert_eq!(*matrix.get(2, 15).unwrap_or(&0.0), 1.0);
}

#[test]
fn test_affine_matrix_2d_identity() {
    // 2D identity: y = [x1, x2] (M=2, N=2)
    let r = 2;
    let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

    let size = 1 << (r * 2); // 2^(2*2) = 16
    assert_eq!(matrix.rows(), size);
    assert_eq!(matrix.cols(), size);
    assert_eq!(matrix.nnz(), size); // Identity

    // Check it's identity
    for i in 0..size {
        assert_eq!(*matrix.get(i, i).unwrap_or(&0.0), 1.0);
    }
}

#[test]
fn test_affine_matrix_2d_swap() {
    // Swap: y1 = x2, y2 = x1 (M=2, N=2)
    let r = 2;
    let params = AffineParams::from_integers(vec![0, 1, 1, 0], vec![0, 0], 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

    let size = 1 << (r * 2); // 16
    assert_eq!(matrix.nnz(), size); // Permutation

    // x_flat = x1 + x2 * 2^R, y_flat = y1 + y2 * 2^R
    // Swap: y1 = x2, y2 = x1, so y_flat = x2 + x1 * 2^R
    let modulus = 1 << r;
    for x1 in 0..modulus {
        for x2 in 0..modulus {
            let x_flat = x1 + x2 * modulus;
            let y_flat = x2 + x1 * modulus; // swapped
            assert_eq!(*matrix.get(y_flat, x_flat).unwrap_or(&0.0), 1.0);
        }
    }
}

#[test]
fn test_affine_matrix_half_scale() {
    // y = x/2, scale=2, R=3, Periodic BC
    // Condition: 2*y ≡ x (mod 2^R=8), so each even x has 2 solutions
    let r = 3;
    let a = vec![Rational64::new(1, 2)];
    let b = vec![Rational64::from_integer(0)];
    let params = AffineParams::new(a, b, 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

    // x=0: y∈{0,4}, x=2: y∈{1,5}, x=4: y∈{2,6}, x=6: y∈{3,7}
    assert_eq!(matrix.nnz(), 8);

    assert_eq!(*matrix.get(0, 0).unwrap_or(&0.0), 1.0);
    assert_eq!(*matrix.get(4, 0).unwrap_or(&0.0), 1.0);
    assert_eq!(*matrix.get(1, 2).unwrap_or(&0.0), 1.0);
    assert_eq!(*matrix.get(5, 2).unwrap_or(&0.0), 1.0);
    assert_eq!(*matrix.get(2, 4).unwrap_or(&0.0), 1.0);
    assert_eq!(*matrix.get(6, 4).unwrap_or(&0.0), 1.0);
    assert_eq!(*matrix.get(3, 6).unwrap_or(&0.0), 1.0);
    assert_eq!(*matrix.get(7, 6).unwrap_or(&0.0), 1.0);

    assert_affine_mpo_matches_matrix(r, &params, &bc);
}

// ========== MPO vs Matrix comparison tests (from Quantics.jl) ==========

fn flat_index_to_site_value(flat: usize, var_count: usize, r: usize, site: usize) -> usize {
    let bit_pos = r - 1 - site;
    (0..var_count)
        .map(|var| {
            let var_value = (flat >> (var * r)) & ((1 << r) - 1);
            ((var_value >> bit_pos) & 1) << var
        })
        .sum()
}

fn column_major_offset(dims: &[usize], coords: &[usize]) -> usize {
    assert_eq!(dims.len(), coords.len());
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&dim, &coord) in dims.iter().zip(coords) {
        assert!(coord < dim);
        offset += coord * stride;
        stride *= dim;
    }
    offset
}

fn affine_matrix_to_dense_tensor(
    matrix: &CsMat<f64>,
    op: &QuanticsOperator,
    r: usize,
    m: usize,
    n: usize,
    template: &TensorDynLen,
) -> TensorDynLen {
    let indices = template.indices().to_vec();
    let dims = template.dims();
    let axis_lookup = tensor_axis_lookup(template);

    let output_positions: Vec<usize> = (0..r)
        .map(|site| {
            let internal_index = &op
                .get_output_mapping(&site)
                .expect("missing output mapping")
                .internal_index;
            tensor_axis_position(&axis_lookup, internal_index)
        })
        .collect();
    let input_positions: Vec<usize> = (0..r)
        .map(|site| {
            let internal_index = &op
                .get_input_mapping(&site)
                .expect("missing input mapping")
                .internal_index;
            tensor_axis_position(&axis_lookup, internal_index)
        })
        .collect();

    let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
    let mut coords = vec![0usize; dims.len()];

    for (y_flat, row) in matrix.outer_iterator().enumerate() {
        for (x_flat, value) in row.iter() {
            coords.fill(0);
            for site in 0..r {
                coords[output_positions[site]] = flat_index_to_site_value(y_flat, m, r, site);
                coords[input_positions[site]] = flat_index_to_site_value(x_flat, n, r, site);
            }
            let offset = column_major_offset(&dims, &coords);
            data[offset] = Complex64::new(*value, 0.0);
        }
    }

    TensorDynLen::from_dense(indices, data).expect("failed to build affine reference tensor")
}

fn affine_interleaved_matrix_to_dense_tensor(
    matrix: &CsMat<f64>,
    op: &QuanticsOperator,
    r: usize,
    m: usize,
    n: usize,
    template: &TensorDynLen,
) -> TensorDynLen {
    let indices = template.indices().to_vec();
    let dims = template.dims();
    let axis_lookup = tensor_axis_lookup(template);

    let output_positions = (0..r)
        .map(|site| {
            op.get_output_mappings(&site)
                .expect("missing output mappings")
                .iter()
                .map(|mapping| tensor_axis_position(&axis_lookup, &mapping.internal_index))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input_positions = (0..r)
        .map(|site| {
            op.get_input_mappings(&site)
                .expect("missing input mappings")
                .iter()
                .map(|mapping| tensor_axis_position(&axis_lookup, &mapping.internal_index))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
    let mut coords = vec![0usize; dims.len()];

    for (y_flat, row) in matrix.outer_iterator().enumerate() {
        for (x_flat, value) in row.iter() {
            coords.fill(0);
            for site in 0..r {
                let bit_pos = r - 1 - site;
                for var in 0..m {
                    let y_var = (y_flat >> (var * r)) & ((1 << r) - 1);
                    coords[output_positions[site][var]] = (y_var >> bit_pos) & 1;
                }
                for var in 0..n {
                    let x_var = (x_flat >> (var * r)) & ((1 << r) - 1);
                    coords[input_positions[site][var]] = (x_var >> bit_pos) & 1;
                }
            }
            let offset = column_major_offset(&dims, &coords);
            data[offset] = Complex64::new(*value, 0.0);
        }
    }

    TensorDynLen::from_dense(indices, data)
        .expect("failed to build interleaved affine reference tensor")
}

/// Assert that the MPO representation matches the direct sparse matrix computation
/// for all elements. This is the primary correctness check: two independent algorithms
/// (carry-based MPO vs direct enumeration) must agree.
const AFFINE_DENSE_REFERENCE_MAX_ELEMENTS: usize = 1 << 20;

fn affine_dense_reference_element_count(r: usize, m: usize, n: usize) -> Option<usize> {
    let local_bits = m.checked_add(n)?;
    let total_bits = r.checked_mul(local_bits)?;
    let shift = u32::try_from(total_bits).ok()?;
    1usize.checked_shl(shift)
}

fn affine_operator_expected_dense_tensor(
    op: &QuanticsOperator,
    template: &TensorDynLen,
    r: usize,
    mut expected: impl FnMut(usize, usize) -> Complex64,
) -> TensorDynLen {
    let indices = template.indices().to_vec();
    let dims = template.dims();
    let axis_lookup = tensor_axis_lookup(template);

    let output_positions: Vec<usize> = (0..r)
        .map(|site| {
            let internal_index = &op
                .get_output_mapping(&site)
                .expect("missing output mapping")
                .internal_index;
            tensor_axis_position(&axis_lookup, internal_index)
        })
        .collect();
    let input_positions: Vec<usize> = (0..r)
        .map(|site| {
            let internal_index = &op
                .get_input_mapping(&site)
                .expect("missing input mapping")
                .internal_index;
            tensor_axis_position(&axis_lookup, internal_index)
        })
        .collect();

    let size = 1usize << r;
    let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
    let mut coords = vec![0usize; dims.len()];
    for x in 0..size {
        for y in 0..size {
            coords.fill(0);
            for site in 0..r {
                let bit_pos = r - 1 - site;
                coords[output_positions[site]] = (y >> bit_pos) & 1;
                coords[input_positions[site]] = (x >> bit_pos) & 1;
            }
            data[column_major_offset(&dims, &coords)] = expected(y, x);
        }
    }

    TensorDynLen::from_dense(indices, data).expect("failed to build expected affine tensor")
}

#[allow(clippy::needless_range_loop)]
fn assert_affine_mpo_matches_matrix(r: usize, params: &AffineParams, bc: &[BoundaryCondition]) {
    let m = params.m();
    let n = params.n();
    let dense_elements = affine_dense_reference_element_count(r, m, n).unwrap_or(usize::MAX);
    assert!(
        dense_elements <= AFFINE_DENSE_REFERENCE_MAX_ELEMENTS,
        "affine dense reference helper is limited to {AFFINE_DENSE_REFERENCE_MAX_ELEMENTS} elements, requested {dense_elements} [r={r}, m={m}, n={n}]"
    );

    let matrix = affine_transform_matrix(r, params, bc).unwrap();
    let op = affine_operator(r, params, bc).unwrap();
    let actual = op.mpo().contract_to_tensor().unwrap();
    let expected = affine_matrix_to_dense_tensor(&matrix, &op, r, m, n, &actual);
    let maxabs = actual.sub(&expected).unwrap().maxabs().unwrap();

    assert!(
        maxabs < 1e-10,
        "MPO vs matrix mismatch: maxabs={} [r={}, m={}, n={}, bc={:?}]",
        maxabs,
        r,
        m,
        n,
        bc
    );
}

#[test]
#[should_panic(expected = "affine dense reference helper is limited")]
fn test_affine_dense_reference_helper_rejects_large_cases() {
    let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];
    assert_affine_mpo_matches_matrix(21, &params, &bc);
}

/// Assert that affine_transform_matrix produces correct results by independently
/// computing y = A*x + b using Rational64 arithmetic (no integer scaling).
/// Equivalent to Julia's test_affine_transform_matrix_multi_variables.
#[allow(clippy::needless_range_loop)]
fn assert_affine_matrix_correctness(r: usize, params: &AffineParams, bc: &[BoundaryCondition]) {
    let m = params.m();
    let n = params.n();
    let modulus = 1i64 << r;

    let matrix = affine_transform_matrix(r, params, bc).unwrap();

    let input_size = 1usize << (r * n);
    let output_size = 1usize << (r * m);

    // Build expected matrix independently using Rational64
    for x_flat in 0..input_size {
        // Decode x_flat to N-dimensional vector
        let x_vals: Vec<i64> = (0..n)
            .map(|var| ((x_flat >> (var * r)) & ((1 << r) - 1)) as i64)
            .collect();

        // Compute y = A*x + b using Rational64 (independent of to_integer_scaled)
        let y_rational: Vec<Rational64> = (0..m)
            .map(|i| {
                let mut val = params.b()[i];
                for j in 0..n {
                    val += params.a()[i + m * j] * Rational64::from_integer(x_vals[j]);
                }
                val
            })
            .collect();

        // Check if all y values are integers
        if y_rational.iter().any(|y| !y.is_integer()) {
            // No valid output for this input - all entries in this column must be 0
            for y_flat in 0..output_size {
                let val = *matrix.get(y_flat, x_flat).unwrap_or(&0.0);
                assert!(
                    val.abs() < 1e-10,
                    "Expected zero at ({}, {}) for non-integer y, got {} [r={}, bc={:?}]",
                    y_flat,
                    x_flat,
                    val,
                    r,
                    bc
                );
            }
            continue;
        }

        let y_int: Vec<i64> = y_rational.iter().map(|y| y.to_integer()).collect();

        // Apply boundary conditions
        let bc_periodic: Vec<bool> = bc
            .iter()
            .map(|b| matches!(b, BoundaryCondition::Periodic))
            .collect();

        let y_bounded: Vec<i64> = y_int
            .iter()
            .enumerate()
            .map(|(i, &yi)| {
                if bc_periodic[i] {
                    ((yi % modulus) + modulus) % modulus
                } else {
                    yi
                }
            })
            .collect();

        let valid = y_bounded
            .iter()
            .enumerate()
            .all(|(i, &yi)| bc_periodic[i] || (yi >= 0 && yi < modulus));

        if valid {
            let y_flat: usize = y_bounded
                .iter()
                .enumerate()
                .map(|(var, &yi)| (yi as usize) << (var * r))
                .sum();

            // This (y_flat, x_flat) should be 1
            let val = *matrix.get(y_flat, x_flat).unwrap_or(&0.0);
            assert!(
                (val - 1.0).abs() < 1e-10,
                "Expected 1 at ({}, {}) but got {} [r={}, x={:?}, y={:?}, bc={:?}]",
                y_flat,
                x_flat,
                val,
                r,
                x_vals,
                y_bounded,
                bc
            );
        }
    }
}

// MPO vs matrix comparison tests

#[test]
fn test_affine_mpo_vs_matrix_1d_identity() {
    let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];
    assert_affine_mpo_matches_matrix(3, &params, &bc);
}

#[test]
fn test_affine_mpo_vs_matrix_1d_shift() {
    let params = AffineParams::from_integers(vec![1], vec![3], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];
    assert_affine_mpo_matches_matrix(3, &params, &bc);
}

#[test]
fn test_affine_mpo_vs_matrix_simple() {
    let params = AffineParams::from_integers(vec![1, 1, 0, 1], vec![0, 0], 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];
    assert_affine_mpo_matches_matrix(3, &params, &bc);
}

#[test]
fn test_affine_matrix_3x3_hard() {
    // From Quantics.jl compare_hard test
    // A = [1 0 1; 1 2 -1; 0 1 1], b = [11; 23; -15]
    let r = 3;
    let a = vec![1i64, 1, 0, 0, 2, 1, 1, -1, 1];
    let b = vec![11i64, 23, -15];
    let params = AffineParams::from_integers(a, b, 3, 3).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 3];
    assert_affine_mpo_matches_matrix(r, &params, &bc);
}

#[test]
fn test_affine_matrix_rectangular() {
    // From Quantics.jl compare_rect test
    // A = [1 0 1; 1 2 0] (2x3), b = [11; -3]
    let r = 4;
    let a = vec![1i64, 1, 0, 2, 1, 0];
    let b = vec![11i64, -3];
    let params = AffineParams::from_integers(a, b, 2, 3).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];
    assert_affine_mpo_matches_matrix(r, &params, &bc);
}

#[test]
fn test_affine_matrix_denom_odd() {
    // From Quantics.jl compare_denom_odd test
    // A = [1/3], b = [0]
    for r in [1, 3, 6] {
        for bc in [BoundaryCondition::Periodic, BoundaryCondition::Open] {
            let a = vec![Rational64::new(1, 3)];
            let b = vec![Rational64::from_integer(0)];
            let params = AffineParams::new(a, b, 1, 1).unwrap();
            let bcs = vec![bc];
            assert_affine_mpo_matches_matrix(r, &params, &bcs);
        }
    }
}

#[test]
fn test_affine_matrix_light_cone() {
    // From Quantics.jl compare_light_cone test
    // Light cone transformation: A = 1/2 * [[1, 1], [1, -1]], b = [2, 3]
    for r in [3, 4] {
        for bc in [BoundaryCondition::Periodic, BoundaryCondition::Open] {
            let a = vec![
                Rational64::new(1, 2),
                Rational64::new(1, 2),
                Rational64::new(1, 2),
                Rational64::new(-1, 2),
            ];
            let b = vec![Rational64::from_integer(2), Rational64::from_integer(3)];
            let params = AffineParams::new(a, b, 2, 2).unwrap();
            let bcs = vec![bc; 2];
            assert_affine_matrix_correctness(r, &params, &bcs);
            assert_affine_mpo_matches_matrix(r, &params, &bcs);
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_affine_matrix_unitarity_full() {
    // From Quantics.jl full test - verify T'*T == I for orthogonal transforms
    // A = [[1, 0], [1, 1]], b = [0, 0]
    let r = 4;
    let a = vec![1i64, 1, 0, 1]; // [[1, 0], [1, 1]] in column-major
    let b = vec![0i64, 0];
    let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let t = affine_transform_matrix(r, &params, &bc).unwrap();

    // Compute T' * T
    let size = 1 << (2 * r);
    let mut prod = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                let t_ki = *t.get(k, i).unwrap_or(&0.0);
                let t_kj = *t.get(k, j).unwrap_or(&0.0);
                sum += t_ki * t_kj;
            }
            prod[i][j] = sum;
        }
    }

    // Check T' * T == I
    for i in 0..size {
        for j in 0..size {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (prod[i][j] - expected).abs() < 1e-10,
                "T'*T not identity at ({}, {}): got {}",
                i,
                j,
                prod[i][j]
            );
        }
    }
}

#[test]
fn test_affine_mpo_vs_matrix_r1() {
    let bc = vec![BoundaryCondition::Periodic];
    // Identity R=1
    let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    assert_affine_mpo_matches_matrix(1, &params, &bc);
    // Shift R=1 (y = x + 1 mod 2)
    let params = AffineParams::from_integers(vec![1], vec![1], 1, 1).unwrap();
    assert_affine_mpo_matches_matrix(1, &params, &bc);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_affine_matrix_unitarity_with_shift() {
    // From Quantics.jl full test with shift - verify T*T' == I
    // A = [[1, 0], [1, 1]], b = [4, 1]
    let r = 4;
    let a = vec![1i64, 1, 0, 1];
    let b = vec![4i64, 1];
    let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let t = affine_transform_matrix(r, &params, &bc).unwrap();

    // Compute T * T'
    let size = 1 << (2 * r);
    let mut prod = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                let t_ik = *t.get(i, k).unwrap_or(&0.0);
                let t_jk = *t.get(j, k).unwrap_or(&0.0);
                sum += t_ik * t_jk;
            }
            prod[i][j] = sum;
        }
    }

    // Check T * T' == I
    for i in 0..size {
        for j in 0..size {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (prod[i][j] - expected).abs() < 1e-10,
                "T*T' not identity at ({}, {}): got {}",
                i,
                j,
                prod[i][j]
            );
        }
    }
}

// ========== Unfused API tests ==========

#[test]
fn test_affine_unfused_basic() {
    // Test unfused API basic functionality
    let r = 3;
    let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Periodic];

    let unfused = affine_transform_tensors_unfused(r, &params, &bc).unwrap();

    assert_eq!(unfused.len(), r);

    // For M=1, N=1, site_dim = 2^2 = 4
    let site_dim = 4;
    // For identity transform y=x, carry is always 0, so bond dimension is 1
    assert_eq!(unfused[0].dims(), &[1, site_dim, 1]); // First tensor: [1, 4, 1]
    assert_eq!(unfused[r - 1].dims(), &[1, site_dim, 1]); // Last tensor: [1, 4, 1]
}

#[test]
fn test_affine_unfused_2d() {
    // Test unfused API with 2D transformation
    let r = 2;
    let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let unfused = affine_transform_tensors_unfused(r, &params, &bc).unwrap();

    assert_eq!(unfused.len(), r);

    // For M=2, N=2, site_dim = 2^4 = 16
    let site_dim = 16;
    for tensor in &unfused {
        assert_eq!(tensor.dims()[1], site_dim);
    }
}

#[test]
fn test_unfused_tensor_info() {
    let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
    let info = UnfusedTensorInfo::new(&params).unwrap();

    assert_eq!(info.m(), 2);
    assert_eq!(info.n(), 2);
    assert_eq!(info.num_physical_dims(), 4);
    assert_eq!(info.physical_dim(), 2);

    // Test shape
    let shape = info.unfused_shape(3, 5).unwrap();
    assert_eq!(shape, vec![3, 2, 2, 2, 2, 5]);

    // Test index encoding/decoding
    // site_idx = y_bits | (x_bits << m)
    // y_bits = y0 + 2*y1, x_bits = x0 + 2*x1
    // Example: y0=1, y1=0, x0=0, x1=1 -> y_bits=1, x_bits=2 -> site_idx = 1 + 4*2 = 9
    let (y_bits, x_bits) = info.decode_fused_index(9).unwrap();
    assert_eq!(y_bits, vec![1, 0]);
    assert_eq!(x_bits, vec![0, 1]);

    let encoded = info.encode_fused_index(&[1, 0], &[0, 1]).unwrap();
    assert_eq!(encoded, 9);
}

#[test]
fn test_unfused_tensor_info_rejects_invalid_bits_and_indices() {
    let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    let info = UnfusedTensorInfo::new(&params).unwrap();

    assert!(info.encode_fused_index(&[], &[0]).is_err());
    assert!(info.encode_fused_index(&[2], &[0]).is_err());
    assert!(info.encode_fused_index(&[0], &[0, 0]).is_err());
    assert!(info.decode_fused_index(4).is_err());

    let malformed = AffineParams {
        a: Vec::new(),
        b: Vec::new(),
        m: usize::MAX,
        n: usize::MAX,
    };
    let error = UnfusedTensorInfo::new(&malformed).unwrap_err();
    assert!(error.to_string().contains("physical dimension count"));
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_unfused_vs_fused_equivalence() {
    // Verify that unfused tensors give the same matrix as fused
    let r = 2;
    let params = AffineParams::from_integers(vec![1, 1, 0, 1], vec![0, 0], 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();
    let unfused = affine_transform_tensors_unfused(r, &params, &bc).unwrap();

    // Contract unfused tensors to matrix
    let info = UnfusedTensorInfo::new(&params).unwrap();
    let m = info.m();
    let n = info.n();
    let output_size = 1 << (m * r);
    let input_size = 1 << (n * r);

    let mut unfused_matrix = vec![vec![Complex64::new(0.0, 0.0); input_size]; output_size];

    for y_flat in 0..output_size {
        for x_flat in 0..input_size {
            let mut left_vec = vec![Complex64::one()];

            for site in 0..r {
                let bit_pos = r - 1 - site;

                let mut y_bits = 0usize;
                for var in 0..m {
                    let y_var = (y_flat >> (var * r)) & ((1 << r) - 1);
                    let bit = (y_var >> bit_pos) & 1;
                    y_bits |= bit << var;
                }

                let mut x_bits = 0usize;
                for var in 0..n {
                    let x_var = (x_flat >> (var * r)) & ((1 << r) - 1);
                    let bit = (x_var >> bit_pos) & 1;
                    x_bits |= bit << var;
                }

                let site_idx = y_bits | (x_bits << m);
                let tensor = &unfused[site];
                let left_dim = tensor.dims()[0];
                let right_dim = tensor.dims()[2];

                let mut new_vec = vec![Complex64::new(0.0, 0.0); right_dim];
                for l in 0..left_dim.min(left_vec.len()) {
                    for rr in 0..right_dim {
                        new_vec[rr] += left_vec[l] * tensor[[l, site_idx, rr]];
                    }
                }
                left_vec = new_vec;
            }

            unfused_matrix[y_flat][x_flat] = if left_vec.is_empty() {
                Complex64::new(0.0, 0.0)
            } else {
                left_vec[0]
            };
        }
    }

    // Compare
    let size = 1 << (2 * r);
    for y in 0..size {
        for x in 0..size {
            let sparse_val = *matrix.get(y, x).unwrap_or(&0.0);
            let unfused_val = unfused_matrix[y][x].re;
            assert!(
                (sparse_val - unfused_val).abs() < 1e-10,
                "Unfused vs fused mismatch at ({}, {}): sparse={}, unfused={}",
                y,
                x,
                sparse_val,
                unfused_val
            );
        }
    }
}

#[test]
fn test_affine_operator_interleaved_matches_matrix() {
    let r = 2;
    let params = AffineParams::from_integers(vec![1, 1, 0, 1], vec![0, 0], 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let matrix = affine_transform_matrix(r, &params, &bc).unwrap();
    let op = affine_operator_interleaved(r, &params, &bc).unwrap();

    for site in 0..r {
        assert_eq!(op.get_output_mappings(&site).unwrap().len(), params.m());
        assert_eq!(op.get_input_mappings(&site).unwrap().len(), params.n());
        for mapping in op.get_output_mappings(&site).unwrap() {
            assert_eq!(mapping.true_index.dim(), 2);
            assert_eq!(mapping.internal_index.dim(), 2);
        }
        for mapping in op.get_input_mappings(&site).unwrap() {
            assert_eq!(mapping.true_index.dim(), 2);
            assert_eq!(mapping.internal_index.dim(), 2);
        }
    }

    let actual = op.mpo().contract_to_tensor().unwrap();
    let expected =
        affine_interleaved_matrix_to_dense_tensor(&matrix, &op, r, params.m(), params.n(), &actual);
    let maxabs = actual.sub(&expected).unwrap().maxabs().unwrap();
    assert!(maxabs < 1e-10, "interleaved affine maxabs={maxabs}");
}

#[test]
fn test_affine_parametric_full() {
    // From Quantics.jl "full R=$R, boundary=$boundary, M=$M, N=$N" test
    struct TestCase {
        a: Vec<i64>,
        b: Vec<i64>,
        m: usize,
        n: usize,
    }

    let cases = vec![
        TestCase {
            a: vec![1],
            b: vec![1],
            m: 1,
            n: 1,
        },
        TestCase {
            a: vec![1, 0],
            b: vec![0],
            m: 1,
            n: 2,
        },
        TestCase {
            a: vec![2, -1],
            b: vec![1],
            m: 1,
            n: 2,
        },
        TestCase {
            a: vec![1, 0],
            b: vec![0, 0],
            m: 2,
            n: 1,
        },
        TestCase {
            a: vec![2, -1],
            b: vec![1, -1],
            m: 2,
            n: 1,
        },
        TestCase {
            a: vec![1, 1, 0, 1],
            b: vec![0, 1],
            m: 2,
            n: 2,
        },
        TestCase {
            a: vec![2, 4, 0, 1],
            b: vec![100, -1],
            m: 2,
            n: 2,
        },
    ];

    for r in [1, 2] {
        for bc_type in [BoundaryCondition::Open, BoundaryCondition::Periodic] {
            for case in &cases {
                let params =
                    AffineParams::from_integers(case.a.clone(), case.b.clone(), case.m, case.n)
                        .unwrap();
                let bc = vec![bc_type; case.m];
                assert_affine_matrix_correctness(r, &params, &bc);
                assert_affine_mpo_matches_matrix(r, &params, &bc);
            }
        }
    }
}

#[test]
fn test_affine_denom_even() {
    // From Quantics.jl compare_denom_even test
    let a = vec![Rational64::new(1, 2)];
    for b_val in [3i64, 5, -3, -5] {
        let b = vec![Rational64::from_integer(b_val)];
        let params = AffineParams::new(a.clone(), b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];
        for r in [3, 5] {
            assert_affine_mpo_matches_matrix(r, &params, &bc);
        }
    }
}

#[test]
fn test_affine_extension_loop() {
    // Test abs(b) >= 2^R with Open BC (requires extension loop)

    // b=[-32, 32] with R=5, identity matrix: abs(32)=2^5=2^R triggers extension
    let r = 5;
    let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![-32, 32], 2, 2).unwrap();
    let bc = vec![BoundaryCondition::Open; 2];
    assert_affine_mpo_matches_matrix(r, &params, &bc);
    assert_affine_matrix_correctness(r, &params, &bc);

    // abs(b) clearly exceeds 2^R: 2^4=16, abs(b)=32 > 16
    let r = 4;
    assert_affine_mpo_matches_matrix(r, &params, &bc);
    assert_affine_matrix_correctness(r, &params, &bc);

    // 1D case: y = x + 64 with R=6, Open BC
    let r = 6;
    let params = AffineParams::from_integers(vec![1], vec![64], 1, 1).unwrap();
    let bc = vec![BoundaryCondition::Open];
    assert_affine_mpo_matches_matrix(r, &params, &bc);
    assert_affine_matrix_correctness(r, &params, &bc);
}

#[test]
fn test_affine_antiperiodic_full_cycle_shifts() {
    let r = 3;
    let n = 1usize << r;

    for (shift, expected_sign) in [(n as i64, -1.0), (-(n as i64), -1.0), (2 * n as i64, 1.0)] {
        let params = AffineParams::from_integers(vec![1], vec![shift], 1, 1).unwrap();
        let matrix =
            affine_transform_matrix(r, &params, &[BoundaryCondition::AntiPeriodic]).unwrap();

        for x in 0..n {
            for y in 0..n {
                let expected = if x == y { expected_sign } else { 0.0 };
                let actual = *matrix.get(y, x).unwrap_or(&0.0);
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "shift={shift}, x={x}, y={y}, expected={expected}, actual={actual}"
                );
            }
        }
    }
}

#[test]
fn test_affine_antiperiodic_full_cycle_plus_one_shift() {
    let r = 3;
    let n = 1usize << r;
    let params = AffineParams::from_integers(vec![1], vec![n as i64 + 1], 1, 1).unwrap();
    let matrix = affine_transform_matrix(r, &params, &[BoundaryCondition::AntiPeriodic]).unwrap();

    for x in 0..n {
        let y_expected = (x + 1) % n;
        let quotient = (x as i64 + n as i64 + 1).div_euclid(n as i64);
        let expected_sign = if quotient.rem_euclid(2) == 0 {
            1.0
        } else {
            -1.0
        };
        for y in 0..n {
            let expected = if y == y_expected { expected_sign } else { 0.0 };
            let actual = *matrix.get(y, x).unwrap_or(&0.0);
            assert!(
                (actual - expected).abs() < 1e-12,
                "x={x}, y={y}, expected={expected}, actual={actual}"
            );
        }
    }
}

#[test]
fn test_affine_antiperiodic_difference_delta_signs() {
    let r = 3;
    let n = 1usize << r;
    let params = AffineParams::from_integers(vec![1, -1], vec![0], 1, 2).unwrap();
    let matrix = affine_transform_matrix(r, &params, &[BoundaryCondition::AntiPeriodic]).unwrap();

    for x in 0..n {
        for xp in 0..n {
            let x_flat = x | (xp << r);
            let z = (x + n - xp) % n;
            let expected_sign = if x >= xp { 1.0 } else { -1.0 };
            for y in 0..n {
                let expected = if y == z { expected_sign } else { 0.0 };
                let actual = *matrix.get(y, x_flat).unwrap_or(&0.0);
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "x={x}, xp={xp}, z={z}, y={y}, expected={expected}, actual={actual}"
                );
            }
        }
    }
}

#[test]
fn test_affine_antiperiodic_mpo_applies_outgoing_carry_parity() {
    let r = 3;
    let modulus = 1i64 << r;
    let params = AffineParams::from_integers(vec![1], vec![1], 1, 1).unwrap();
    let operator = affine_operator(r, &params, &[BoundaryCondition::AntiPeriodic]).unwrap();
    let actual = operator.mpo().contract_to_tensor().unwrap();
    let expected = affine_operator_expected_dense_tensor(&operator, &actual, r, |y, x| {
        let raw = x as i64 + 1;
        let expected_y = raw.rem_euclid(modulus) as usize;
        let expected_sign = if raw.div_euclid(modulus).rem_euclid(2) == 0 {
            1.0
        } else {
            -1.0
        };
        Complex64::new(if y == expected_y { expected_sign } else { 0.0 }, 0.0)
    });
    let maxabs = actual.sub(&expected).unwrap().maxabs().unwrap();
    assert!(maxabs < 1e-12, "anti-periodic MPO mismatch: {maxabs}");
}
