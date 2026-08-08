//! C API for canonical QTT layouts and transform materialization.

use crate::types::{
    t4a_boundary_condition, t4a_qtt_layout, t4a_qtt_layout_kind, t4a_treetn, InternalIndex,
    InternalQttLayout, InternalTreeTN,
};
use crate::{
    capi_error, clone_opaque, err_null_pointer, is_assigned_opaque, release_opaque, run_catching,
    t4a_status_code, CapiResult, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER,
};
use num_complex::Complex64;
use num_rational::Rational64;
use tensor4all_core::{IndexLike, TensorDynLen};
use tensor4all_quanticstransform::{
    affine_operator, cumsum_operator, flip_operator, phase_rotation_operator,
    quantics_fourier_operator, shift_operator, AffineParams, BoundaryCondition, FourierOptions,
};

/// Release a QTT layout handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtt_layout_release(obj: *mut t4a_qtt_layout) {
    release_opaque(obj);
}

/// Clone a QTT layout handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtt_layout_clone(
    src: *const t4a_qtt_layout,
    out: *mut *mut t4a_qtt_layout,
) -> t4a_status_code {
    clone_opaque(src, out)
}

/// Check whether a QTT layout handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtt_layout_is_assigned(obj: *const t4a_qtt_layout) -> i32 {
    is_assigned_opaque(obj)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChainSiteKind {
    Single,
    First,
    Middle,
    Last,
}

#[derive(Clone)]
struct SourceSite {
    kind: ChainSiteKind,
    left_dim: usize,
    out_dim: usize,
    in_dim: usize,
    right_dim: usize,
    data: Vec<Complex64>,
}

impl SourceSite {
    fn from_chain_treetn(tn: &InternalTreeTN, site: usize, nsites: usize) -> CapiResult<Self> {
        let node_idx = tn
            .node_index(&site)
            .ok_or_else(|| capi_error(T4A_INVALID_ARGUMENT, format!("missing node {site}")))?;
        let tensor = tn.tensor(node_idx).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("missing tensor at node {site}"),
            )
        })?;
        let dims = tensor.dims();
        let data = tensor
            .to_vec::<Complex64>()
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        let (kind, left_dim, out_dim, in_dim, right_dim) = match (site == 0, site + 1 == nsites) {
            (true, true) => {
                if dims.len() != 2 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        format!(
                            "single-site operator tensor must have rank 2, got {}",
                            dims.len()
                        ),
                    ));
                }
                (ChainSiteKind::Single, 1, dims[0], dims[1], 1)
            }
            (true, false) => {
                if dims.len() != 3 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        format!("first operator tensor must have rank 3, got {}", dims.len()),
                    ));
                }
                (ChainSiteKind::First, 1, dims[0], dims[1], dims[2])
            }
            (false, true) => {
                if dims.len() != 3 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        format!("last operator tensor must have rank 3, got {}", dims.len()),
                    ));
                }
                (ChainSiteKind::Last, dims[0], dims[1], dims[2], 1)
            }
            (false, false) => {
                if dims.len() != 4 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        format!(
                            "middle operator tensor must have rank 4, got {}",
                            dims.len()
                        ),
                    ));
                }
                (ChainSiteKind::Middle, dims[0], dims[1], dims[2], dims[3])
            }
        };

        Ok(Self {
            kind,
            left_dim,
            out_dim,
            in_dim,
            right_dim,
            data,
        })
    }

    fn value(&self, left: usize, out: usize, input: usize, right: usize) -> Complex64 {
        match self.kind {
            ChainSiteKind::Single => self.data[out + self.out_dim * input],
            ChainSiteKind::First => self.data[out + self.out_dim * (input + self.in_dim * right)],
            ChainSiteKind::Middle => {
                self.data
                    [left + self.left_dim * (out + self.out_dim * (input + self.in_dim * right))]
            }
            ChainSiteKind::Last => self.data[left + self.left_dim * (out + self.out_dim * input)],
        }
    }
}

fn require_layout_or_status<'a>(
    layout: *const t4a_qtt_layout,
) -> std::result::Result<&'a InternalQttLayout, t4a_status_code> {
    if layout.is_null() {
        return Err(err_null_pointer("layout"));
    }
    Ok(unsafe { (&*layout).inner() })
}

macro_rules! require_layout_or_return {
    ($layout:expr) => {{
        match require_layout_or_status($layout) {
            Ok(layout) => layout,
            Err(code) => {
                return code;
            }
        }
    }};
}

fn bit_dim(bits: usize, what: &str) -> CapiResult<usize> {
    let shift = u32::try_from(bits).map_err(|_| {
        capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{what} exceeds the usize shift width: {bits}"),
        )
    })?;
    1usize
        .checked_shl(shift)
        .ok_or_else(|| capi_error(T4A_INVALID_ARGUMENT, format!("{what} is too large: {bits}")))
}

fn checked_allocation_len<T>(dims: &[usize], name: &str) -> CapiResult<usize> {
    let elements = dims.iter().try_fold(1usize, |product, &dim| {
        product.checked_mul(dim).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{name} element count overflows usize for dimensions {dims:?}"),
            )
        })
    })?;
    let element_size = std::mem::size_of::<T>();
    if element_size != 0 {
        let bytes = elements.checked_mul(element_size).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{name} byte length overflows usize for dimensions {dims:?}"),
            )
        })?;
        if bytes > isize::MAX as usize {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{name} byte length exceeds isize::MAX for dimensions {dims:?}"),
            ));
        }
    }
    Ok(elements)
}

fn checked_slice_len<T>(name: &str, len: usize) -> CapiResult<()> {
    let element_size = std::mem::size_of::<T>();
    if element_size != 0 && len > isize::MAX as usize / element_size {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{name} byte length exceeds isize::MAX"),
        ));
    }
    Ok(())
}

fn try_vec_with_capacity<T>(name: &str, capacity: usize) -> CapiResult<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|err| {
        capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{name} allocation failed for capacity {capacity}: {err}"),
        )
    })?;
    Ok(values)
}

fn validate_single_var_materialization_layout(
    layout: &InternalQttLayout,
    target_var: usize,
) -> CapiResult<usize> {
    if target_var >= layout.nvariables() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "target_var must be smaller than nvariables",
        ));
    }

    let r = layout.resolution(target_var);

    if layout.kind() == t4a_qtt_layout_kind::Fused {
        let phys_dim = bit_dim(layout.nvariables(), "fused local dimension")?;
        checked_allocation_len::<Complex64>(&[phys_dim, phys_dim], "fused site tensor")?;
    }

    Ok(r)
}

fn validate_affine_materialization_layout(layout: &InternalQttLayout) -> CapiResult<()> {
    checked_allocation_len::<SourceSite>(&[layout.nsites()], "affine site list")?;
    checked_allocation_len::<TensorDynLen>(&[layout.nsites()], "affine tensor list")?;
    checked_allocation_len::<InternalIndex>(
        &[layout.nsites().saturating_sub(1)],
        "affine bond index list",
    )?;
    Ok(())
}

fn extract_chain_sites(mpo: &InternalTreeTN) -> CapiResult<Vec<SourceSite>> {
    let nsites = mpo.node_count();
    if nsites == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "materialized operator must contain at least one site",
        ));
    }

    let mut sites = try_vec_with_capacity::<SourceSite>("source site list", nsites)?;
    for site in 0..nsites {
        sites.push(SourceSite::from_chain_treetn(mpo, site, nsites)?);
    }
    Ok(sites)
}

fn build_chain_tensor<F>(
    left_bond: Option<InternalIndex>,
    out_index: InternalIndex,
    in_index: InternalIndex,
    right_bond: Option<InternalIndex>,
    eval: F,
) -> CapiResult<TensorDynLen>
where
    F: Fn(usize, usize, usize, usize) -> Complex64,
{
    let left_dim = left_bond.as_ref().map_or(1, |idx| idx.dim());
    let out_dim = out_index.dim();
    let in_dim = in_index.dim();
    let right_dim = right_bond.as_ref().map_or(1, |idx| idx.dim());

    let mut indices = try_vec_with_capacity::<InternalIndex>("chain tensor indices", 4)?;
    let mut dims = try_vec_with_capacity::<usize>("chain tensor dimensions", 4)?;
    let has_left = left_bond.is_some();
    let has_right = right_bond.is_some();

    if let Some(idx) = left_bond {
        dims.push(left_dim);
        indices.push(idx);
    }
    dims.push(out_dim);
    indices.push(out_index);
    dims.push(in_dim);
    indices.push(in_index);
    if let Some(idx) = right_bond {
        dims.push(right_dim);
        indices.push(idx);
    }

    let total_size = checked_allocation_len::<Complex64>(&dims, "chain tensor")?;
    let mut data = try_vec_with_capacity::<Complex64>("chain tensor data", total_size)?;
    data.resize(total_size, Complex64::new(0.0, 0.0));

    for left in 0..left_dim {
        for out in 0..out_dim {
            for input in 0..in_dim {
                for right in 0..right_dim {
                    let flat = match (has_left, has_right) {
                        (false, false) => out + out_dim * input,
                        (false, true) => out + out_dim * (input + in_dim * right),
                        (true, false) => left + left_dim * (out + out_dim * input),
                        (true, true) => {
                            left + left_dim * (out + out_dim * (input + in_dim * right))
                        }
                    };
                    data[flat] = eval(left, out, input, right);
                }
            }
        }
    }

    TensorDynLen::from_dense(indices, data).map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))
}

fn build_identity_site(
    left_bond: Option<InternalIndex>,
    out_index: InternalIndex,
    in_index: InternalIndex,
    right_bond: Option<InternalIndex>,
) -> CapiResult<TensorDynLen> {
    let left_dim = left_bond.as_ref().map_or(1, |idx| idx.dim());
    let right_dim = right_bond.as_ref().map_or(1, |idx| idx.dim());
    if left_dim != right_dim {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!(
                "identity bridge requires matching bond dimensions, got {left_dim} and {right_dim}"
            ),
        ));
    }

    build_chain_tensor(
        left_bond,
        out_index,
        in_index,
        right_bond,
        |left, out, input, right| {
            if left == right && out == input {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        },
    )
}

fn build_treetn_from_chain(tensors: Vec<TensorDynLen>) -> CapiResult<InternalTreeTN> {
    let mut node_names = try_vec_with_capacity::<usize>("chain node-name list", tensors.len())?;
    node_names.extend(0..tensors.len());
    InternalTreeTN::from_tensors(tensors, node_names)
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))
}

fn single_var_positions(layout: &InternalQttLayout, target_var: usize) -> CapiResult<Vec<usize>> {
    if target_var >= layout.nvariables() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "target_var must be smaller than nvariables",
        ));
    }

    let resolution = layout.resolution(target_var);
    let positions = match layout.kind() {
        t4a_qtt_layout_kind::Interleaved => {
            let nvariables = layout.nvariables();
            let mut positions =
                try_vec_with_capacity::<usize>("single-variable source positions", resolution)?;
            for level in 0..resolution {
                positions.push(level * nvariables + target_var);
            }
            positions
        }
        t4a_qtt_layout_kind::Fused => {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "single_var_positions is not valid for fused layouts",
            ))
        }
    };
    Ok(positions)
}

fn expand_chain_with_identities(
    source_sites: &[SourceSite],
    nsites: usize,
    source_positions: &[usize],
) -> CapiResult<InternalTreeTN> {
    if source_sites.is_empty() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "source operator must contain at least one site",
        ));
    }
    if source_sites.len() != source_positions.len() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "source_positions length does not match source operator length",
        ));
    }
    if source_positions.iter().any(|&pos| pos >= nsites) {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "source_positions must all be smaller than nsites",
        ));
    }
    if source_positions
        .windows(2)
        .any(|window| window[0] >= window[1])
    {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "source_positions must be strictly increasing",
        ));
    }

    let phys_dim = source_sites[0].out_dim;
    if source_sites
        .iter()
        .any(|site| site.out_dim != phys_dim || site.in_dim != phys_dim)
    {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "expanded identity embedding requires square per-site dimensions",
        ));
    }

    let bond_count = nsites.saturating_sub(1);
    let mut bond_dims = try_vec_with_capacity::<usize>("expanded bond list", bond_count)?;
    bond_dims.resize(bond_count, 1);
    for (src_idx, window) in source_positions.windows(2).enumerate() {
        let bond_dim = source_sites[src_idx].right_dim;
        for dim in bond_dims.iter_mut().take(window[1]).skip(window[0]) {
            *dim = bond_dim;
        }
    }
    let mut bond_indices =
        try_vec_with_capacity::<InternalIndex>("expanded bond indices", bond_dims.len())?;
    for &dim in &bond_dims {
        bond_indices.push(InternalIndex::new_dyn(dim));
    }

    let mut tensors = try_vec_with_capacity::<TensorDynLen>("expanded tensor list", nsites)?;
    let mut next_source = 0usize;
    for site in 0..nsites {
        let left_bond = (site > 0).then(|| bond_indices[site - 1].clone());
        let right_bond = (site + 1 < nsites).then(|| bond_indices[site].clone());
        let out_index = InternalIndex::new_dyn(phys_dim);
        let in_index = InternalIndex::new_dyn(phys_dim);

        let tensor =
            if next_source < source_positions.len() && source_positions[next_source] == site {
                let src = &source_sites[next_source];
                next_source += 1;
                build_chain_tensor(
                    left_bond,
                    out_index,
                    in_index,
                    right_bond,
                    |left, out, input, right| src.value(left, out, input, right),
                )?
            } else {
                build_identity_site(left_bond, out_index, in_index, right_bond)?
            };
        tensors.push(tensor);
    }

    build_treetn_from_chain(tensors)
}

fn embed_single_var_fused(
    layout: &InternalQttLayout,
    target_var: usize,
    source_sites: &[SourceSite],
) -> CapiResult<InternalTreeTN> {
    let nvariables = layout.nvariables();
    if target_var >= nvariables {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "target_var must be smaller than nvariables",
        ));
    }
    let phys_dim = bit_dim(nvariables, "fused local dimension")?;
    checked_allocation_len::<Complex64>(&[phys_dim, phys_dim], "fused site tensor")?;
    let nsites = layout.nsites();
    if source_sites.len() != nsites {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "source operator length does not match fused layout length",
        ));
    }

    let bond_count = source_sites.len().saturating_sub(1);
    let mut bond_indices =
        try_vec_with_capacity::<InternalIndex>("fused bond indices", bond_count)?;
    for site in source_sites.iter().take(bond_count) {
        bond_indices.push(InternalIndex::new_dyn(site.right_dim));
    }

    let mut tensors = try_vec_with_capacity::<TensorDynLen>("fused tensor list", nsites)?;
    for (site_idx, src) in source_sites.iter().enumerate() {
        let left_bond = (site_idx > 0).then(|| bond_indices[site_idx - 1].clone());
        let right_bond = (site_idx + 1 < nsites).then(|| bond_indices[site_idx].clone());
        let out_index = InternalIndex::new_dyn(phys_dim);
        let in_index = InternalIndex::new_dyn(phys_dim);

        tensors.push(build_chain_tensor(
            left_bond,
            out_index,
            in_index,
            right_bond,
            |left, out_multi, in_multi, right| {
                for variable in 0..nvariables {
                    if variable == target_var {
                        continue;
                    }
                    if ((out_multi >> variable) & 1) != ((in_multi >> variable) & 1) {
                        return Complex64::new(0.0, 0.0);
                    }
                }
                let out_bit = (out_multi >> target_var) & 1;
                let in_bit = (in_multi >> target_var) & 1;
                src.value(left, out_bit, in_bit, right)
            },
        )?);
    }

    build_treetn_from_chain(tensors)
}

fn materialize_single_var_operator(
    layout: &InternalQttLayout,
    target_var: usize,
    source_mpo: InternalTreeTN,
) -> CapiResult<InternalTreeTN> {
    let source_sites = extract_chain_sites(&source_mpo)?;
    match layout.kind() {
        t4a_qtt_layout_kind::Interleaved => {
            let positions = single_var_positions(layout, target_var)?;
            if positions.len() != source_sites.len() {
                return Err(capi_error(
                    T4A_INVALID_ARGUMENT,
                    "layout resolution does not match source operator length",
                ));
            }
            expand_chain_with_identities(&source_sites, layout.nsites(), &positions)
        }
        t4a_qtt_layout_kind::Fused => embed_single_var_fused(layout, target_var, &source_sites),
    }
}

fn rational_from_i64_parts(
    name: &str,
    index: usize,
    numerator: i64,
    denominator: i64,
) -> CapiResult<Rational64> {
    if denominator == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{name}[{index}] has zero denominator"),
        ));
    }

    let numerator_magnitude = numerator.unsigned_abs();
    let denominator_magnitude = denominator.unsigned_abs();
    let mut a = numerator_magnitude;
    let mut b = denominator_magnitude;
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    let gcd = a.max(1);
    let numerator_magnitude = numerator_magnitude / gcd;
    let denominator_magnitude = denominator_magnitude / gcd;
    if denominator_magnitude > i64::MAX as u64 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{name}[{index}] denominator is not representable after reduction"),
        ));
    }

    let negative = (numerator < 0) ^ (denominator < 0);
    let numerator = if numerator_magnitude == 1u64 << 63 {
        if negative {
            i64::MIN
        } else {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{name}[{index}] numerator is not representable after reduction"),
            ));
        }
    } else {
        let magnitude = numerator_magnitude as i64;
        if negative {
            -magnitude
        } else {
            magnitude
        }
    };
    Ok(Rational64::new(numerator, denominator_magnitude as i64))
}

fn parse_rationals(
    numerators: *const i64,
    denominators: *const i64,
    len: usize,
    name: &str,
) -> CapiResult<Vec<Rational64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if numerators.is_null() || denominators.is_null() {
        return Err(capi_error(
            T4A_NULL_POINTER,
            format!("{name} numerator/denominator array is null"),
        ));
    }
    checked_slice_len::<i64>(name, len)?;
    let mut values = try_vec_with_capacity::<Rational64>(name, len)?;

    let nums = unsafe { std::slice::from_raw_parts(numerators, len) };
    let dens = unsafe { std::slice::from_raw_parts(denominators, len) };
    for (i, (&num, &den)) in nums.iter().zip(dens.iter()).enumerate() {
        values.push(rational_from_i64_parts(name, i, num, den)?);
    }
    Ok(values)
}

fn parse_boundary_conditions(
    bc: *const t4a_boundary_condition,
    len: usize,
) -> CapiResult<Vec<BoundaryCondition>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if bc.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "bc is null"));
    }
    checked_slice_len::<t4a_boundary_condition>("bc", len)?;
    let mut values = try_vec_with_capacity::<BoundaryCondition>("boundary-condition list", len)?;
    for &value in unsafe { std::slice::from_raw_parts(bc, len) } {
        values.push(value.into());
    }
    Ok(values)
}

/// Create an immutable canonical QTT layout descriptor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtt_layout_new(
    kind: t4a_qtt_layout_kind,
    nvariables: usize,
    variable_resolutions: *const usize,
    out: *mut *mut t4a_qtt_layout,
) -> t4a_status_code {
    run_catching(out, || {
        if nvariables == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "nvariables must be greater than zero",
            ));
        }
        if variable_resolutions.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "variable_resolutions is null"));
        }
        checked_slice_len::<usize>("variable_resolutions", nvariables)?;

        let resolutions = unsafe { std::slice::from_raw_parts(variable_resolutions, nvariables) };
        let mut resolution_values =
            try_vec_with_capacity::<usize>("variable resolution list", nvariables)?;
        resolution_values.extend_from_slice(resolutions);
        let layout = InternalQttLayout::new(kind, resolution_values)
            .map_err(|msg| capi_error(T4A_INVALID_ARGUMENT, msg))?;
        Ok(t4a_qtt_layout::new(layout))
    })
}

/// Materialize a shift transform directly as a chain-shaped TreeTN.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_shift_materialize(
    layout: *const t4a_qtt_layout,
    target_var: usize,
    offset: i64,
    bc: t4a_boundary_condition,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let layout_ref = require_layout_or_return!(layout);

    run_catching(out, || {
        let r = validate_single_var_materialization_layout(layout_ref, target_var)?;
        let source = shift_operator(r, offset, bc.into())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let treetn = materialize_single_var_operator(layout_ref, target_var, source.into_mpo())?;
        Ok(t4a_treetn::new(treetn))
    })
}

/// Materialize a flip transform directly as a chain-shaped TreeTN.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_flip_materialize(
    layout: *const t4a_qtt_layout,
    target_var: usize,
    bc: t4a_boundary_condition,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let layout_ref = require_layout_or_return!(layout);

    run_catching(out, || {
        let r = validate_single_var_materialization_layout(layout_ref, target_var)?;
        let source =
            flip_operator(r, bc.into()).map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let treetn = materialize_single_var_operator(layout_ref, target_var, source.into_mpo())?;
        Ok(t4a_treetn::new(treetn))
    })
}

/// Materialize a phase-rotation transform directly as a chain-shaped TreeTN.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_phase_rotation_materialize(
    layout: *const t4a_qtt_layout,
    target_var: usize,
    theta: f64,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let layout_ref = require_layout_or_return!(layout);

    run_catching(out, || {
        let r = validate_single_var_materialization_layout(layout_ref, target_var)?;
        let source = phase_rotation_operator(r, theta)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let treetn = materialize_single_var_operator(layout_ref, target_var, source.into_mpo())?;
        Ok(t4a_treetn::new(treetn))
    })
}

/// Materialize a cumulative-sum transform directly as a chain-shaped TreeTN.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_cumsum_materialize(
    layout: *const t4a_qtt_layout,
    target_var: usize,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let layout_ref = require_layout_or_return!(layout);

    run_catching(out, || {
        let r = validate_single_var_materialization_layout(layout_ref, target_var)?;
        let source = cumsum_operator(r).map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let treetn = materialize_single_var_operator(layout_ref, target_var, source.into_mpo())?;
        Ok(t4a_treetn::new(treetn))
    })
}

/// Materialize a Fourier transform directly as a chain-shaped TreeTN.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_fourier_materialize(
    layout: *const t4a_qtt_layout,
    target_var: usize,
    forward: i32,
    maxbonddim: usize,
    tolerance: f64,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let layout_ref = require_layout_or_return!(layout);

    run_catching(out, || {
        let r = validate_single_var_materialization_layout(layout_ref, target_var)?;
        let mut options = if forward != 0 {
            FourierOptions::forward()
        } else {
            FourierOptions::inverse()
        };
        if maxbonddim > 0 {
            options.maxbonddim = maxbonddim;
        }
        if tolerance > 0.0 {
            options.tolerance = tolerance;
        }
        let source = quantics_fourier_operator(r, options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let treetn = materialize_single_var_operator(layout_ref, target_var, source.into_mpo())?;
        Ok(t4a_treetn::new(treetn))
    })
}

/// Materialize the forward affine operator `y = A * x + b` as a chain-shaped
/// TreeTN using the Fused QTT layout.
///
/// `a_num[i + k * m]` and `a_den[i + k * m]` hold the numerator and
/// denominator of `A[i, k]` (column-major, length `m * n`, where `i`
/// is the row index 0..m and `k` is the column index 0..n). `b_num[i]`
/// and `b_den[i]` describe the `i`-th component of `b` (length `m`).
/// `bc[i]` is the boundary condition applied to output coordinate `i`.
/// The resulting TreeTN has `layout->nsites()` nodes, each with fused
/// input and output site indices of dimensions `2^n` and `2^m`
/// respectively.
///
/// To obtain the pullback operator `f(y) = g(A * y + b)`, materialize the
/// forward operator with this function and transpose at the binding layer
/// (the pullback is exactly the transpose of the forward operator).
///
/// # Errors
///
/// Returns `T4A_INVALID_ARGUMENT` if `m == 0`, `n == 0`, `m * n`, the
/// layout-dependent site-list allocation, or a fused site/allocation size
/// overflows, `layout->kind()` is not `Fused`, `b_den[i] == 0`, or
/// `a_den[i + k * m] == 0`. Dimension and byte checks run before reading any
/// coefficient or boundary-condition pointer, so invalid layout dimensions
/// take precedence over null coefficient and boundary-condition pointers.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_affine_materialize(
    layout: *const t4a_qtt_layout,
    a_num: *const i64,
    a_den: *const i64,
    b_num: *const i64,
    b_den: *const i64,
    m: usize,
    n: usize,
    bc: *const t4a_boundary_condition,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let layout_ref = require_layout_or_return!(layout);

    run_catching(out, || {
        if m == 0 || n == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "affine materialization requires m > 0 and n > 0",
            ));
        }
        if layout_ref.kind() != t4a_qtt_layout_kind::Fused {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "affine materialization currently supports fused layouts only",
            ));
        }
        validate_affine_materialization_layout(layout_ref)?;

        let a_len = m.checked_mul(n).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("affine matrix element count overflows usize: {m} × {n}"),
            )
        })?;
        checked_slice_len::<i64>("a", a_len)?;
        checked_slice_len::<i64>("b", m)?;
        let input_dim = bit_dim(n, "affine input local dimension")?;
        let output_dim = bit_dim(m, "affine output local dimension")?;
        checked_allocation_len::<Complex64>(&[output_dim, input_dim], "affine fused site tensor")?;

        let a = parse_rationals(a_num, a_den, a_len, "a")?;
        let b = parse_rationals(b_num, b_den, m, "b")?;
        let bc = parse_boundary_conditions(bc, m)?;
        let params =
            AffineParams::new(a, b, m, n).map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let source = affine_operator(layout_ref.nsites(), &params, &bc)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(source.into_mpo()))
    })
}

#[cfg(test)]
mod tests;
