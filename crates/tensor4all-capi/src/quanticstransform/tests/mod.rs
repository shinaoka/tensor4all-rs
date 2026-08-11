use super::*;
use crate::{t4a_index, t4a_treetn_release, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use num_complex::Complex64;
use num_rational::Rational64;
use tensor4all_core::TensorDynLen;
use tensor4all_quanticstransform::{
    affine_operator, cumsum_operator, flip_operator, phase_rotation_operator,
    quantics_fourier_operator, shift_operator, AffineParams, BoundaryCondition, FourierOptions,
};
use tensor4all_treetn::LinearOperator;

fn new_layout(kind: t4a_qtt_layout_kind, resolutions: &[usize]) -> *mut t4a_qtt_layout {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_qtt_layout_new(kind, resolutions.len(), resolutions.as_ptr(), &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn last_error() -> String {
    let mut len = 0usize;
    assert_eq!(
        crate::t4a_last_error_message(std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut buf = vec![0u8; len];
    assert_eq!(
        crate::t4a_last_error_message(buf.as_mut_ptr(), buf.len(), &mut len),
        T4A_SUCCESS
    );
    std::ffi::CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

fn decode_mixed_radix(mut flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut values = Vec::with_capacity(dims.len());
    for &dim in dims {
        values.push(flat % dim);
        flat /= dim;
    }
    values
}

fn rust_operator_matrix(
    op: &LinearOperator<TensorDynLen, usize>,
    out_dims: &[usize],
    in_dims: &[usize],
) -> Vec<Complex64> {
    let nsites = out_dims.len();
    assert_eq!(in_dims.len(), nsites);
    let nrows: usize = out_dims.iter().product();
    let ncols: usize = in_dims.iter().product();

    let mut indices = Vec::with_capacity(2 * nsites);
    for site in 0..nsites {
        indices.push(
            op.get_output_mappings(&site)
                .unwrap()
                .first()
                .unwrap()
                .internal_index
                .clone(),
        );
        indices.push(
            op.get_input_mappings(&site)
                .unwrap()
                .first()
                .unwrap()
                .internal_index
                .clone(),
        );
    }

    let mut matrix = vec![Complex64::new(0.0, 0.0); nrows * ncols];
    for x in 0..ncols {
        let in_values = decode_mixed_radix(x, in_dims);
        for y in 0..nrows {
            let out_values = decode_mixed_radix(y, out_dims);
            let mut values = Vec::with_capacity(2 * nsites);
            for site in 0..nsites {
                values.push(out_values[site]);
                values.push(in_values[site]);
            }
            let result = op.mpo().evaluate_point(&indices, &values).unwrap();
            matrix[y + nrows * x] = result.into();
        }
    }
    matrix
}

fn c_operator_matrix(
    op: *const t4a_treetn,
    out_dims: &[usize],
    in_dims: &[usize],
) -> Vec<Complex64> {
    let nsites = out_dims.len();
    assert_eq!(in_dims.len(), nsites);

    let mut n_vertices = 0usize;
    assert_eq!(
        crate::treetn::t4a_treetn_num_vertices(op, &mut n_vertices),
        T4A_SUCCESS
    );
    assert_eq!(n_vertices, nsites);

    let mut out_indices = Vec::with_capacity(nsites);
    let mut in_indices = Vec::with_capacity(nsites);
    for site in 0..nsites {
        let mut len = 0usize;
        assert_eq!(
            crate::treetn::t4a_treetn_site_indices(op, site, std::ptr::null_mut(), 0, &mut len),
            T4A_SUCCESS
        );
        assert_eq!(len, 2);
        let mut pair = vec![std::ptr::null_mut(); len];
        assert_eq!(
            crate::treetn::t4a_treetn_site_indices(
                op,
                site,
                pair.as_mut_ptr(),
                pair.len(),
                &mut len
            ),
            T4A_SUCCESS
        );

        let mut dim_out = 0usize;
        let mut dim_in = 0usize;
        assert_eq!(
            crate::index::t4a_index_dim(pair[0], &mut dim_out),
            T4A_SUCCESS
        );
        assert_eq!(
            crate::index::t4a_index_dim(pair[1], &mut dim_in),
            T4A_SUCCESS
        );
        assert_eq!(dim_out, out_dims[site]);
        assert_eq!(dim_in, in_dims[site]);

        out_indices.push(pair[0]);
        in_indices.push(pair[1]);
    }

    let nrows: usize = out_dims.iter().product();
    let ncols: usize = in_dims.iter().product();
    let mut matrix = vec![Complex64::new(0.0, 0.0); nrows * ncols];
    let ordered_indices: Vec<*const t4a_index> = (0..nsites)
        .flat_map(|site| {
            [
                out_indices[site] as *const t4a_index,
                in_indices[site] as *const t4a_index,
            ]
        })
        .collect();

    for x in 0..ncols {
        let in_values = decode_mixed_radix(x, in_dims);
        for y in 0..nrows {
            let out_values = decode_mixed_radix(y, out_dims);
            let mut values = Vec::with_capacity(2 * nsites);
            for site in 0..nsites {
                values.push(out_values[site]);
                values.push(in_values[site]);
            }
            let mut re = 0.0;
            let mut im = 0.0;
            assert_eq!(
                crate::treetn::t4a_treetn_evaluate(
                    op,
                    ordered_indices.as_ptr(),
                    ordered_indices.len(),
                    values.as_ptr(),
                    1,
                    &mut re,
                    &mut im
                ),
                T4A_SUCCESS
            );
            matrix[y + nrows * x] = Complex64::new(re, im);
        }
    }

    for index in out_indices.into_iter().chain(in_indices) {
        crate::index::t4a_index_release(index);
    }
    matrix
}

fn assert_matrix_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (slot, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (*a - *e).norm();
        assert!(
            err < 1e-10,
            "matrix mismatch at slot {slot}: actual={a:?}, expected={e:?}, err={err}"
        );
    }
}

#[test]
fn test_flip_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_flip_materialize(layout, 0, t4a_boundary_condition::Periodic, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(
        &flip_operator(2, BoundaryCondition::Periodic).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_phase_rotation_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_phase_rotation_materialize(layout, 0, std::f64::consts::PI / 4.0, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(
        &phase_rotation_operator(2, std::f64::consts::PI / 4.0).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_cumsum_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_cumsum_materialize(layout, 0, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(&cumsum_operator(2).unwrap(), &[2, 2], &[2, 2]);
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_fourier_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_fourier_materialize(layout, 0, 1, 8, 1e-12, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let mut options = FourierOptions::forward();
    options.max_bond_dim = 8;
    options.tolerance = 1e-12;
    let expected = rust_operator_matrix(
        &quantics_fourier_operator(2, options).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_qtt_layout_clone_assignment_and_inverse_fourier_match_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    assert_eq!(t4a_qtt_layout_is_assigned(layout), 1);
    assert_eq!(t4a_qtt_layout_is_assigned(std::ptr::null()), 0);

    let mut clone = std::ptr::null_mut();
    assert_eq!(t4a_qtt_layout_clone(layout, &mut clone), T4A_SUCCESS);
    assert!(!clone.is_null());

    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_fourier_materialize(clone, 0, 0, 0, 0.0, &mut op),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(
        &quantics_fourier_operator(2, FourierOptions::inverse()).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);

    t4a_treetn_release(op);
    t4a_qtt_layout_release(clone);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_affine_fused_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    let a_num = [1i64];
    let a_den = [1i64];
    let b_num = [1i64];
    let b_den = [1i64];
    let bc = [t4a_boundary_condition::Periodic];
    assert_eq!(
        t4a_qtransform_affine_materialize(
            layout,
            a_num.as_ptr(),
            a_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let params = AffineParams::new(
        vec![Rational64::from_integer(1)],
        vec![Rational64::from_integer(1)],
        1,
        1,
    )
    .unwrap();
    let expected = rust_operator_matrix(
        &affine_operator(2, &params, &[BoundaryCondition::Periodic]).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);

    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_affine_fused_materialization_handles_i64_extreme_coefficients() {
    for &(coefficient, translation) in &[
        (i64::MIN, i64::MIN),
        (i64::MIN, i64::MAX),
        (i64::MAX, i64::MIN),
        (i64::MAX, i64::MAX),
    ] {
        let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
        let a_num = [coefficient];
        let a_den = [1i64];
        let b_num = [translation];
        let b_den = [1i64];
        let bc = [t4a_boundary_condition::Periodic];
        let mut op = std::ptr::null_mut();
        assert_eq!(
            t4a_qtransform_affine_materialize(
                layout,
                a_num.as_ptr(),
                a_den.as_ptr(),
                b_num.as_ptr(),
                b_den.as_ptr(),
                1,
                1,
                bc.as_ptr(),
                &mut op,
            ),
            T4A_SUCCESS
        );
        let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
        // The C matrix helper enumerates the two MPO sites in chain order,
        // while the affine formula uses integer values with site 0 as MSB.
        let reverse_bits = |value: usize| ((value & 1) << 1) | ((value >> 1) & 1);
        for x_site in 0..4usize {
            let x_value = reverse_bits(x_site) as i128;
            let expected_y =
                ((coefficient as i128) * x_value + translation as i128).rem_euclid(4) as usize;
            for row_site in 0..4usize {
                let expected = if reverse_bits(row_site) == expected_y {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert_eq!(
                    actual[row_site + 4 * x_site],
                    expected,
                    "matrix entry ({row_site}, {x_site}) for a={coefficient}, b={translation}"
                );
            }
        }
        t4a_treetn_release(op);
        t4a_qtt_layout_release(layout);
    }
}

#[test]
fn test_try_vec_with_capacity_reports_invalid_argument_context() {
    let (status, error) = try_vec_with_capacity::<u8>("test C site list", usize::MAX).unwrap_err();
    assert_eq!(status, T4A_INVALID_ARGUMENT);
    assert!(error.contains("test C site list"));
    assert!(
        error.contains("allocation failed"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_public_phase_materializer_reservation_failure_preserves_sentinel() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[usize::MAX]);
    let sentinel = std::ptr::NonNull::<t4a_treetn>::dangling().as_ptr();
    let mut op = sentinel;

    assert_eq!(
        t4a_qtransform_phase_rotation_materialize(layout, 0, 0.0, &mut op),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(op, sentinel);
    let error = last_error();
    assert!(
        error.contains("allocation failed"),
        "expected actual reservation failure, got: {error}"
    );

    t4a_qtt_layout_release(layout);
}

#[test]
fn test_fused_allocation_checks_f64_and_c64_byte_boundaries() {
    let f64_elements = isize::MAX as usize / std::mem::size_of::<f64>() + 1;
    let c64_elements = isize::MAX as usize / std::mem::size_of::<Complex64>() + 1;
    let f64_error = checked_allocation_len::<f64>(&[f64_elements], "f64 site").unwrap_err();
    let c64_error = checked_allocation_len::<Complex64>(&[c64_elements], "c64 site").unwrap_err();
    assert_eq!(f64_error.0, T4A_INVALID_ARGUMENT);
    assert_eq!(c64_error.0, T4A_INVALID_ARGUMENT);
    assert!(f64_error.1.contains("byte length"));
    assert!(c64_error.1.contains("byte length"));
}

#[test]
fn test_checked_allocation_len_covers_product_and_zero_size_branches() {
    let product_error = checked_allocation_len::<u8>(&[usize::MAX, 2], "product").unwrap_err();
    assert_eq!(product_error.0, T4A_INVALID_ARGUMENT);
    assert!(product_error.1.contains("element count"));
    assert_eq!(
        checked_allocation_len::<u8>(&[0, usize::MAX], "zero").unwrap(),
        0
    );
    assert_eq!(checked_allocation_len::<u8>(&[], "empty").unwrap(), 1);
    assert_eq!(
        checked_allocation_len::<()>(&[usize::MAX], "zst").unwrap(),
        usize::MAX
    );
}

#[test]
fn test_fused_materializers_reject_oversized_site_list_before_source_mpo() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[usize::MAX]);
    let mut op = std::ptr::null_mut();

    macro_rules! assert_rejected {
        ($call:expr) => {{
            assert_eq!($call, T4A_INVALID_ARGUMENT);
            assert!(op.is_null());
            op = std::ptr::null_mut();
            let error = last_error();
            assert!(
                error.contains("site") || error.contains("allocation") || error.contains("byte"),
                "unexpected diagnostic: {error}"
            );
        }};
    }

    assert_rejected!(t4a_qtransform_flip_materialize(
        layout,
        0,
        t4a_boundary_condition::Periodic,
        &mut op
    ));
    assert_rejected!(t4a_qtransform_phase_rotation_materialize(
        layout, 0, 0.0, &mut op
    ));
    assert_rejected!(t4a_qtransform_cumsum_materialize(layout, 0, &mut op));
    assert_rejected!(t4a_qtransform_fourier_materialize(
        layout, 0, 1, 0, 0.0, &mut op
    ));
    assert_rejected!(t4a_qtransform_shift_materialize(
        layout,
        0,
        0,
        t4a_boundary_condition::Periodic,
        &mut op
    ));
    assert!(op.is_null());

    t4a_qtt_layout_release(layout);
}

#[test]
fn test_affine_layout_site_allocation_precedes_coefficient_pointer_parsing() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[usize::MAX]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_affine_materialize(
            layout,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            1,
            std::ptr::null(),
            &mut op,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(op.is_null());
    let error = last_error();
    assert!(
        error.contains("site") || error.contains("allocation") || error.contains("byte"),
        "unexpected diagnostic: {error}"
    );
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_affine_rational_parts_handle_i64_extremes_without_panic() {
    let value = rational_from_i64_parts("a", 0, i64::MIN, 1).unwrap();
    assert_eq!(value, Rational64::from_integer(i64::MIN));

    let error = rational_from_i64_parts("a", 0, i64::MIN, -1).unwrap_err();
    assert!(error.1.contains("not representable"));
}

#[test]
fn test_fused_multivar_site_dimension_errors_are_reported() {
    let resolutions = vec![1usize; usize::BITS as usize / 2];
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &resolutions);
    let mut op = std::ptr::null_mut();

    assert_eq!(
        t4a_qtransform_shift_materialize(layout, 0, 0, t4a_boundary_condition::Periodic, &mut op,),
        T4A_INVALID_ARGUMENT
    );
    let error = last_error();
    assert!(
        error.contains("site") || error.contains("byte") || error.contains("allocation"),
        "unexpected diagnostic: {error}"
    );
    assert!(op.is_null());

    t4a_qtt_layout_release(layout);
}

#[test]
fn test_layout_and_affine_validation_errors_are_reported() {
    let mut layout = std::ptr::null_mut();
    assert_eq!(
        t4a_qtt_layout_new(t4a_qtt_layout_kind::Fused, 0, std::ptr::null(), &mut layout),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("nvariables must be greater than zero"));

    assert_eq!(
        t4a_qtt_layout_new(
            t4a_qtt_layout_kind::Fused,
            usize::MAX,
            std::ptr::NonNull::<usize>::dangling().as_ptr(),
            &mut layout,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("byte length"));

    let resolutions = [2usize];
    assert_eq!(
        t4a_qtt_layout_new(
            t4a_qtt_layout_kind::Fused,
            resolutions.len(),
            std::ptr::null(),
            &mut layout
        ),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("variable_resolutions is null"));

    let fused = new_layout(t4a_qtt_layout_kind::Fused, &resolutions);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_shift_materialize(fused, 1, 0, t4a_boundary_condition::Periodic, &mut op),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("target_var must be smaller than nvariables"));

    assert_eq!(
        t4a_qtransform_shift_materialize(
            std::ptr::null(),
            0,
            0,
            t4a_boundary_condition::Periodic,
            &mut op
        ),
        T4A_NULL_POINTER
    );
    let err = last_error();
    assert!(err.contains("layout") || err.contains("null"), "{err}");

    let interleaved = new_layout(t4a_qtt_layout_kind::Interleaved, &resolutions);
    let a_num = [1i64];
    let a_den = [1i64];
    let b_num = [0i64];
    let b_den = [1i64];
    let bc = [t4a_boundary_condition::Periodic];
    assert_eq!(
        t4a_qtransform_affine_materialize(
            interleaved,
            a_num.as_ptr(),
            a_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("fused layouts only"));

    let zero_den = [0i64];
    assert_eq!(
        t4a_qtransform_affine_materialize(
            fused,
            a_num.as_ptr(),
            zero_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("zero denominator"));

    assert_eq!(
        t4a_qtransform_affine_materialize(
            fused,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            usize::MAX,
            2,
            std::ptr::null(),
            &mut op,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("element count") || last_error().contains("dimension"));

    assert_eq!(
        t4a_qtransform_affine_materialize(
            fused,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            1,
            bc.as_ptr(),
            &mut op,
        ),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("a numerator/denominator array is null"));

    t4a_qtt_layout_release(interleaved);
    t4a_qtt_layout_release(fused);
}

/// Build the expected 16×16 dense matrix for an interleaved 2-variable 2-bit layout
/// where the single-var 2-site operator is embedded at `var_sites` and identity occupies
/// the complementary sites.
///
/// `var_sites[level]` gives the chain position for the variable being acted on.
/// The complementary sites carry identity.
///
/// Index ordering (column-major / little-endian):
///   row = Σ_k  out_k * 2^k   (k = 0..4)
///   col = Σ_k  in_k  * 2^k
///
/// For var0 (sites 0, 2) acting on a 4-site chain:
///   var_sites = [0, 2]   identity_sites = [1, 3]
/// For var1 (sites 1, 3):
///   var_sites = [1, 3]   identity_sites = [0, 2]
fn expected_interleaved_shift_matrix(
    shift_4x4: &[Complex64],
    var_sites: [usize; 2],
) -> Vec<Complex64> {
    // identity_sites are the two sites NOT in var_sites (within 0..4)
    let identity_sites: Vec<usize> = (0..4).filter(|s| !var_sites.contains(s)).collect();
    let is0 = identity_sites[0];
    let is1 = identity_sites[1];

    // shift_4x4 is indexed (row, col) where
    //   row = out_lvl0 + 2*out_lvl1   (mixed-radix in level order)
    //   col = in_lvl0  + 2*in_lvl1
    let mut expected = vec![Complex64::new(0.0, 0.0); 16 * 16];
    // Enumerate all 2^4 = 16 output states and 16 input states.
    for col in 0..16usize {
        // Decode column → per-site input bits
        let i: [usize; 4] = std::array::from_fn(|k| (col >> k) & 1);
        for row in 0..16usize {
            // Decode row → per-site output bits
            let o: [usize; 4] = std::array::from_fn(|k| (row >> k) & 1);

            // Identity condition: identity sites must pass through unchanged
            if o[is0] != i[is0] || o[is1] != i[is1] {
                continue;
            }

            // Operator element for the variable sites:
            //   shift_4x4 row = out_lvl0 + 2*out_lvl1  (var bit at level 0, then level 1)
            //   shift_4x4 col = in_lvl0  + 2*in_lvl1
            let shift_row = o[var_sites[0]] + 2 * o[var_sites[1]];
            let shift_col = i[var_sites[0]] + 2 * i[var_sites[1]];
            expected[row + 16 * col] = shift_4x4[shift_row + 4 * shift_col];
        }
    }
    expected
}

#[test]
fn test_shift_interleaved_multivar_materialization_matches_rust_reference() {
    // 2 variables × 2 bits each → 4 sites in interleaved order:
    //   site 0 = var0_lvl0, site 1 = var1_lvl0, site 2 = var0_lvl1, site 3 = var1_lvl1
    // We shift var0 (target_var = 0) by 1 with Periodic BC.
    let layout = new_layout(t4a_qtt_layout_kind::Interleaved, &[2, 2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_shift_materialize(layout, 0, 1, t4a_boundary_condition::Periodic, &mut op),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2, 2, 2], &[2, 2, 2, 2]);

    // Reference: 2-site shift-by-1 on var0 (sites 0 and 2); identity on var1 (sites 1 and 3).
    let shift_4x4 = rust_operator_matrix(
        &shift_operator(2, 1, BoundaryCondition::Periodic).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    // var0 occupies interleaved positions [0, 2]
    let expected = expected_interleaved_shift_matrix(&shift_4x4, [0, 2]);

    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_shift_interleaved_second_var_matches_rust_reference() {
    // Same layout but we shift var1 (target_var = 1).
    // var1 occupies interleaved positions [1, 3]; identity on var0 at [0, 2].
    let layout = new_layout(t4a_qtt_layout_kind::Interleaved, &[2, 2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_shift_materialize(layout, 1, 1, t4a_boundary_condition::Periodic, &mut op),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2, 2, 2], &[2, 2, 2, 2]);

    // Reference: 2-site shift-by-1 on var1 (sites 1 and 3); identity on var0 (sites 0 and 2).
    let shift_4x4 = rust_operator_matrix(
        &shift_operator(2, 1, BoundaryCondition::Periodic).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    // var1 occupies interleaved positions [1, 3]
    let expected = expected_interleaved_shift_matrix(&shift_4x4, [1, 3]);

    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}
