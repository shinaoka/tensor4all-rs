//! C API for the reduced TreeTN surface.

use crate::types::{
    t4a_canonical_form, t4a_contract_method, t4a_factorize_alg, t4a_index, t4a_scalar_kind,
    t4a_svd_truncation_policy, t4a_tensor, t4a_treetn, t4a_treetn_evaluator, InternalIndex,
    InternalTreeTN, InternalTreeTNEvaluatorHandle,
};
use crate::{
    capi_error, clone_opaque, is_assigned_opaque, panic_message, release_opaque, run_catching,
    set_last_error, t4a_status_code, CapiResult, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR,
    T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_core::{AnyScalar, ColMajorArrayRef, IndexLike, SvdTruncationPolicy};
use tensor4all_treetn::contraction::{self, ContractionMethod, ContractionOptions};
use tensor4all_treetn::{
    apply_linear_operator, partial_contract, square_linsolve, ApplyOptions, CachedEvaluatorOptions,
    CanonicalForm, CanonicalizationOptions, IndexMapping, LinearOperator, LinsolveOptions,
    PartialContractionSpec, RestructureOptions, SiteIndexNetwork, SplitOptions, SwapOptions,
    TreeTNCachedEvaluator, TruncationOptions,
};

/// Release a TreeTN handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_release(obj: *mut t4a_treetn) {
    release_opaque(obj);
}

/// Clone a TreeTN handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_clone(
    src: *const t4a_treetn,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    clone_opaque(src, out)
}

/// Check whether a TreeTN handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_is_assigned(obj: *const t4a_treetn) -> i32 {
    is_assigned_opaque(obj)
}

/// Release a reusable TreeTN evaluator handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluator_release(obj: *mut t4a_treetn_evaluator) {
    release_opaque(obj);
}

/// Clone a reusable TreeTN evaluator handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluator_clone(
    src: *const t4a_treetn_evaluator,
    out: *mut *mut t4a_treetn_evaluator,
) -> t4a_status_code {
    clone_opaque(src, out)
}

/// Check whether a reusable TreeTN evaluator handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluator_is_assigned(obj: *const t4a_treetn_evaluator) -> i32 {
    is_assigned_opaque(obj)
}

#[inline]
fn resolve_svd_policy(policy: *const t4a_svd_truncation_policy) -> Option<SvdTruncationPolicy> {
    if policy.is_null() {
        None
    } else {
        Some(SvdTruncationPolicy::from(unsafe { *policy }))
    }
}

#[inline]
fn resolve_qr_rtol(qr_rtol: f64) -> Option<f64> {
    (qr_rtol != 0.0).then_some(qr_rtol)
}

#[inline]
fn resolve_convergence_tol(convergence_tol: f64) -> Option<f64> {
    (convergence_tol > 0.0).then_some(convergence_tol)
}

#[inline]
fn resolve_fit_nfullsweeps(nfullsweeps: usize) -> usize {
    if nfullsweeps == 0 {
        1
    } else {
        nfullsweeps
    }
}

fn orthogonalize_error_message(force: libc::c_int, err: impl std::fmt::Display) -> String {
    let mut msg = format!("{err:#}");
    if force == 0 {
        msg = msg.replace("CanonicalizationOptions::forced()", "force=1");
        if msg == "canonicalize: form mismatch" {
            msg.push_str(
                ": The network is already canonicalized with a different form. \
                 Pass force=1 to re-canonicalize with a different form.",
            );
        }
    }
    msg
}

fn run_status<F>(f: F) -> t4a_status_code
where
    F: FnOnce() -> CapiResult<()>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => T4A_SUCCESS,
        Ok(Err((code, msg))) => {
            set_last_error(&msg);
            code
        }
        Err(panic) => {
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            T4A_INTERNAL_ERROR
        }
    }
}

fn require_tree<'a>(ptr: *const t4a_treetn) -> CapiResult<&'a t4a_treetn> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "treetn is null"));
    }
    Ok(unsafe { &*ptr })
}

fn require_tree_mut<'a>(ptr: *mut t4a_treetn) -> CapiResult<&'a mut t4a_treetn> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "treetn is null"));
    }
    Ok(unsafe { &mut *ptr })
}

fn require_evaluator_mut<'a>(
    ptr: *mut t4a_treetn_evaluator,
) -> CapiResult<&'a mut t4a_treetn_evaluator> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "treetn evaluator is null"));
    }
    Ok(unsafe { &mut *ptr })
}

fn require_node(tn: &InternalTreeTN, vertex: usize) -> CapiResult<()> {
    if tn.node_index(&vertex).is_none() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("vertex {vertex} does not exist"),
        ));
    }
    Ok(())
}

fn query_then_fill_copy<T: Copy>(
    values: &[T],
    buf: *mut T,
    buf_len: usize,
    out_len: *mut usize,
    what: &str,
) -> CapiResult<()> {
    if out_len.is_null() {
        return Err(capi_error(
            T4A_NULL_POINTER,
            format!("{what}: out_len is null"),
        ));
    }

    unsafe { *out_len = values.len() };
    if buf.is_null() {
        return Ok(());
    }
    checked_ptr_span::<T>(values.len(), what)?;
    if buf_len < values.len() {
        return Err(capi_error(
            T4A_BUFFER_TOO_SMALL,
            format!(
                "{what}: buffer too small, need {}, got {}",
                values.len(),
                buf_len
            ),
        ));
    }

    for (i, value) in values.iter().enumerate() {
        unsafe { *buf.add(i) = *value };
    }
    Ok(())
}

fn checked_ptr_span<T>(len: usize, what: &str) -> CapiResult<()> {
    if len > (isize::MAX as usize) / std::mem::size_of::<T>() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{what} byte span overflows isize"),
        ));
    }
    Ok(())
}

fn collect_indices(
    index_ptrs: *const *const t4a_index,
    n_indices: usize,
    what: &str,
) -> CapiResult<Vec<InternalIndex>> {
    checked_ptr_span::<*const t4a_index>(n_indices, what)?;
    if n_indices == 0 {
        return Ok(Vec::new());
    }
    if index_ptrs.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{what} is null")));
    }

    let mut indices = Vec::with_capacity(n_indices);
    for i in 0..n_indices {
        let ptr = unsafe { *index_ptrs.add(i) };
        if ptr.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, format!("{what}[{i}] is null")));
        }
        indices.push(unsafe { &*ptr }.inner().clone());
    }
    Ok(indices)
}

fn collect_index_pairs(
    left_ptrs: *const *const t4a_index,
    right_ptrs: *const *const t4a_index,
    n_pairs: usize,
    what: &str,
) -> CapiResult<Vec<(InternalIndex, InternalIndex)>> {
    let left = collect_indices(left_ptrs, n_pairs, &format!("{what}.left"))?;
    let right = collect_indices(right_ptrs, n_pairs, &format!("{what}.right"))?;
    Ok(left.into_iter().zip(right).collect())
}

fn collect_positions(
    positions: *const libc::size_t,
    len: usize,
    what: &str,
) -> CapiResult<Vec<usize>> {
    checked_ptr_span::<libc::size_t>(len, what)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    if positions.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{what} is null")));
    }

    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(unsafe { *positions.add(i) });
    }
    Ok(out)
}

fn collect_edge_endpoints(
    sources: *const libc::size_t,
    targets: *const libc::size_t,
    n_edges: usize,
    what: &str,
) -> CapiResult<Vec<(usize, usize)>> {
    let sources = collect_positions(sources, n_edges, &format!("{what}.sources"))?;
    let targets = collect_positions(targets, n_edges, &format!("{what}.targets"))?;
    Ok(sources.into_iter().zip(targets).collect())
}

fn write_evaluation_results(
    results: &[AnyScalar],
    scalar_kind: t4a_scalar_kind,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> CapiResult<()> {
    checked_ptr_span::<libc::c_double>(results.len(), "out_re")?;
    if !out_im.is_null() {
        checked_ptr_span::<libc::c_double>(results.len(), "out_im")?;
    }
    if out_re.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "out_re is null"));
    }

    match scalar_kind {
        t4a_scalar_kind::F64 => {
            for (i, scalar) in results.iter().enumerate() {
                unsafe { *out_re.add(i) = scalar.real() };
                if !out_im.is_null() {
                    unsafe { *out_im.add(i) = 0.0 };
                }
            }
        }
        t4a_scalar_kind::C64 => {
            if out_im.is_null() {
                return Err(capi_error(
                    T4A_NULL_POINTER,
                    "out_im is required for complex-valued evaluation results",
                ));
            }
            for (i, scalar) in results.iter().enumerate() {
                let z: Complex64 = scalar.clone().into();
                unsafe {
                    *out_re.add(i) = z.re;
                    *out_im.add(i) = z.im;
                }
            }
        }
    }
    Ok(())
}

fn treetn_scalar_kind(tn: &InternalTreeTN) -> t4a_scalar_kind {
    if tn
        .node_indices()
        .into_iter()
        .any(|node| tn.tensor(node).is_some_and(|tensor| tensor.is_complex()))
    {
        t4a_scalar_kind::C64
    } else {
        t4a_scalar_kind::F64
    }
}

fn collect_target_assignment(
    assignment_site_indices: *const *const t4a_index,
    assignment_target_vertices: *const libc::size_t,
    n_assignments: usize,
    what: &str,
) -> CapiResult<HashMap<InternalIndex, usize>> {
    let site_indices = collect_indices(
        assignment_site_indices,
        n_assignments,
        &format!("{what}.site_indices"),
    )?;
    let target_vertices = collect_positions(
        assignment_target_vertices,
        n_assignments,
        &format!("{what}.target_vertices"),
    )?;
    let mut target_assignment = HashMap::with_capacity(n_assignments);
    for (siteind, target_vertex) in site_indices.into_iter().zip(target_vertices) {
        if let Some(previous_target) = target_assignment.insert(siteind.clone(), target_vertex) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "{what}: duplicate site index assignment for {:?}: {} and {}",
                    siteind, previous_target, target_vertex
                ),
            ));
        }
    }
    Ok(target_assignment)
}

#[allow(clippy::too_many_arguments)]
fn build_target_network(
    target_vertices: *const libc::size_t,
    n_target_vertices: usize,
    target_site_indices: *const *const t4a_index,
    target_site_indices_len: *const libc::size_t,
    target_edge_sources: *const libc::size_t,
    target_edge_targets: *const libc::size_t,
    n_target_edges: usize,
    what: &str,
) -> CapiResult<SiteIndexNetwork<usize, InternalIndex>> {
    if n_target_vertices == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{what}: target must contain at least one vertex"),
        ));
    }

    let vertices = collect_positions(
        target_vertices,
        n_target_vertices,
        &format!("{what}.vertices"),
    )?;
    let siteind_lens = collect_positions(
        target_site_indices_len,
        n_target_vertices,
        &format!("{what}.siteind_lens"),
    )?;
    let total_site_indices = siteind_lens.iter().try_fold(0usize, |total, &len| {
        total.checked_add(len).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{what}: total site-index count overflows size_t"),
            )
        })
    })?;
    let flat_site_indices = collect_indices(
        target_site_indices,
        total_site_indices,
        &format!("{what}.site_indices"),
    )?;
    let edges = collect_edge_endpoints(
        target_edge_sources,
        target_edge_targets,
        n_target_edges,
        &format!("{what}.edges"),
    )?;

    let mut network = SiteIndexNetwork::with_capacity(n_target_vertices, n_target_edges);
    let mut seen_site_indices: HashMap<InternalIndex, usize> =
        HashMap::with_capacity(total_site_indices);
    let mut offset = 0usize;
    for (slot, &vertex) in vertices.iter().enumerate() {
        let len = siteind_lens[slot];
        if len == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{what}: target vertex {vertex} must contain at least one site index"),
            ));
        }
        let mut site_space = HashSet::with_capacity(len);
        let end = offset.checked_add(len).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{what}: site-index slice end overflows size_t"),
            )
        })?;
        for siteind in &flat_site_indices[offset..end] {
            if let Some(previous_vertex) = seen_site_indices.insert(siteind.clone(), vertex) {
                return Err(capi_error(
                    T4A_INVALID_ARGUMENT,
                    format!(
                        "{what}: duplicate site index {:?} appears in target vertices {} and {}",
                        siteind, previous_vertex, vertex
                    ),
                ));
            }
            site_space.insert(siteind.clone());
        }
        offset = end;
        network
            .add_node(vertex, site_space)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, format!("{what}: {err}")))?;
    }

    for (source, target) in edges {
        network
            .add_edge(&source, &target)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, format!("{what}: {err}")))?;
    }

    Ok(network)
}

fn build_split_options(
    policy: *const t4a_svd_truncation_policy,
    maxdim: usize,
    final_sweep: libc::c_int,
) -> SplitOptions {
    let mut options = SplitOptions::new()
        .with_form(CanonicalForm::Unitary)
        .with_final_sweep(final_sweep != 0);
    if maxdim > 0 {
        options = options.with_max_bond_dim(maxdim);
    }
    if let Some(policy) = resolve_svd_policy(policy) {
        options = options.with_svd_policy(policy);
    }
    options
}

fn build_swap_options(maxdim: usize, rtol: f64) -> SwapOptions {
    SwapOptions {
        max_bond_dim: (maxdim > 0).then_some(maxdim),
        rtol: (rtol > 0.0).then_some(rtol),
    }
}

fn build_final_truncation(
    policy: *const t4a_svd_truncation_policy,
    maxdim: usize,
) -> Option<TruncationOptions> {
    let policy = resolve_svd_policy(policy);
    if maxdim == 0 && policy.is_none() {
        return None;
    }

    let mut options = TruncationOptions::new();
    if maxdim > 0 {
        options = options.with_max_bond_dim(maxdim);
    }
    if let Some(policy) = policy {
        options = options.with_svd_policy(policy);
    }
    Some(options)
}

fn require_chain_layout(tn: &InternalTreeTN, what: &str) -> CapiResult<usize> {
    let n = tn.node_count();
    if n == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{what} must not be empty"),
        ));
    }
    if tn.edge_count() != n.saturating_sub(1) {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!(
                "{what} must be a chain with {} edges, got {}",
                n.saturating_sub(1),
                tn.edge_count()
            ),
        ));
    }

    for position in 0..n {
        if tn.node_index(&position).is_none() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "{what} must use dense node names 0..{}, missing {position}",
                    n - 1
                ),
            ));
        }

        let mut neighbors: Vec<usize> = tn.site_index_network().neighbors(&position).collect();
        neighbors.sort_unstable();
        let mut expected = Vec::with_capacity(2);
        if position > 0 {
            expected.push(position - 1);
        }
        if position + 1 < n {
            expected.push(position + 1);
        }
        if neighbors != expected {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "{what} must follow chain adjacency at node {position}, got neighbors {:?}, expected {:?}",
                    neighbors, expected
                ),
            ));
        }
    }

    Ok(n)
}

fn collect_single_site_indices(tn: &InternalTreeTN, what: &str) -> CapiResult<Vec<InternalIndex>> {
    let n = require_chain_layout(tn, what)?;
    let mut site_indices = Vec::with_capacity(n);
    for position in 0..n {
        let site_space = tn.site_space(&position).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{what} node {position} has no site index"),
            )
        })?;
        if site_space.len() != 1 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "{what} node {position} must have exactly one site index, got {}",
                    site_space.len()
                ),
            ));
        }
        let site_index = site_space.iter().next().cloned().ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{what} node {position} has no site index"),
            )
        })?;
        site_indices.push(site_index);
    }
    Ok(site_indices)
}

fn validate_mapped_positions(
    mapped_positions: &[usize],
    state_nsites: usize,
    operator_nsites: usize,
) -> CapiResult<()> {
    if mapped_positions.is_empty() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "mapped node positions must not be empty",
        ));
    }
    if mapped_positions.len() != operator_nsites {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!(
                "mapped node positions length {} does not match operator node count {}",
                mapped_positions.len(),
                operator_nsites
            ),
        ));
    }
    for (i, &position) in mapped_positions.iter().enumerate() {
        if position >= state_nsites {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "mapped node position {position} is outside the state chain with {state_nsites} nodes"
                ),
            ));
        }
        if i > 0 && mapped_positions[i - 1] >= position {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "mapped node positions must be strictly increasing",
            ));
        }
    }
    Ok(())
}

fn remap_operator_nodes_to_positions(
    operator: &InternalTreeTN,
    mapped_positions: &[usize],
) -> CapiResult<InternalTreeTN> {
    let operator_nsites = require_chain_layout(operator, "operator")?;
    if operator_nsites != mapped_positions.len() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!(
                "operator node count {} does not match mapped node positions length {}",
                operator_nsites,
                mapped_positions.len()
            ),
        ));
    }

    let mut tensors = Vec::with_capacity(operator_nsites);
    for position in 0..operator_nsites {
        let node_idx = operator.node_index(&position).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator must use dense node names 0..{}, missing {position}",
                    operator_nsites - 1
                ),
            )
        })?;
        let tensor = operator.tensor(node_idx).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("operator node {position} has no tensor"),
            )
        })?;
        tensors.push(tensor.clone());
    }

    InternalTreeTN::from_tensors(tensors, mapped_positions.to_vec())
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))
}

fn build_chain_linear_operator(
    operator: &InternalTreeTN,
    state_true_inputs: &[InternalIndex],
    mapped_positions: &[usize],
    internal_inputs: &[InternalIndex],
    internal_outputs: &[InternalIndex],
    true_outputs: &[InternalIndex],
) -> CapiResult<LinearOperator<crate::types::InternalTensor, usize>> {
    let remapped_operator = remap_operator_nodes_to_positions(operator, mapped_positions)?;
    let mut input_mapping = HashMap::with_capacity(mapped_positions.len());
    let mut output_mapping = HashMap::with_capacity(mapped_positions.len());

    for (slot, &position) in mapped_positions.iter().enumerate() {
        let site_space = remapped_operator.site_space(&position).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("operator node {position} has no site indices"),
            )
        })?;
        if site_space.len() != 2 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator node {position} must have exactly two site indices, got {}",
                    site_space.len()
                ),
            ));
        }

        let internal_input = &internal_inputs[slot];
        let internal_output = &internal_outputs[slot];
        if !site_space.contains(internal_input) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator node {position} does not contain the provided internal input index"
                ),
            ));
        }
        if !site_space.contains(internal_output) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator node {position} does not contain the provided internal output index"
                ),
            ));
        }

        let true_input = &state_true_inputs[position];
        let true_output = &true_outputs[slot];
        if true_input.dim() != internal_input.dim() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "input index shape mismatch at node {position}: state has {}, operator internal input has {}",
                    true_input.dim(),
                    internal_input.dim()
                ),
            ));
        }
        if true_output.dim() != internal_output.dim() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "output index shape mismatch at node {position}: requested true output has {}, operator internal output has {}",
                    true_output.dim(),
                    internal_output.dim()
                ),
            ));
        }

        input_mapping.insert(
            position,
            IndexMapping {
                true_index: true_input.clone(),
                internal_index: internal_input.clone(),
            },
        );
        output_mapping.insert(
            position,
            IndexMapping {
                true_index: true_output.clone(),
                internal_index: internal_output.clone(),
            },
        );
    }

    Ok(LinearOperator::new(
        remapped_operator,
        input_mapping,
        output_mapping,
    ))
}

type LinsolveMappings = (
    HashMap<usize, IndexMapping<InternalIndex>>,
    HashMap<usize, IndexMapping<InternalIndex>>,
);

#[allow(clippy::too_many_arguments)]
fn build_linsolve_index_mappings(
    operator: &InternalTreeTN,
    rhs: &InternalTreeTN,
    init: &InternalTreeTN,
    mapped_vertices: *const libc::size_t,
    n_mapped_vertices: usize,
    true_input_indices: *const *const t4a_index,
    internal_input_indices: *const *const t4a_index,
    true_output_indices: *const *const t4a_index,
    internal_output_indices: *const *const t4a_index,
) -> CapiResult<Option<LinsolveMappings>> {
    if n_mapped_vertices == 0 {
        return Ok(None);
    }

    let mapped_vertices = collect_positions(mapped_vertices, n_mapped_vertices, "mapped_vertices")?;
    let true_inputs = collect_indices(true_input_indices, n_mapped_vertices, "true_input_indices")?;
    let internal_inputs = collect_indices(
        internal_input_indices,
        n_mapped_vertices,
        "internal_input_indices",
    )?;
    let true_outputs = collect_indices(
        true_output_indices,
        n_mapped_vertices,
        "true_output_indices",
    )?;
    let internal_outputs = collect_indices(
        internal_output_indices,
        n_mapped_vertices,
        "internal_output_indices",
    )?;

    let mut input_mapping = HashMap::with_capacity(n_mapped_vertices);
    let mut output_mapping = HashMap::with_capacity(n_mapped_vertices);

    for slot in 0..n_mapped_vertices {
        let vertex = mapped_vertices[slot];
        require_node(operator, vertex)?;
        require_node(rhs, vertex)?;
        require_node(init, vertex)?;

        let operator_site_space = operator.site_space(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("operator vertex {vertex} has no site indices"),
            )
        })?;
        let rhs_site_space = rhs.site_space(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("rhs vertex {vertex} has no site indices"),
            )
        })?;
        let init_site_space = init.site_space(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("init vertex {vertex} has no site indices"),
            )
        })?;

        let true_input = &true_inputs[slot];
        let internal_input = &internal_inputs[slot];
        let true_output = &true_outputs[slot];
        let internal_output = &internal_outputs[slot];

        if !init_site_space.contains(true_input) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("init vertex {vertex} does not contain the provided true input index"),
            ));
        }
        if !rhs_site_space.contains(true_output) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("rhs vertex {vertex} does not contain the provided true output index"),
            ));
        }
        if !operator_site_space.contains(internal_input) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator vertex {vertex} does not contain the provided internal input index"
                ),
            ));
        }
        if !operator_site_space.contains(internal_output) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator vertex {vertex} does not contain the provided internal output index"
                ),
            ));
        }
        if true_input.dim() != internal_input.dim() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "input index shape mismatch at vertex {vertex}: init has {}, operator internal input has {}",
                    true_input.dim(),
                    internal_input.dim()
                ),
            ));
        }
        if true_output.dim() != internal_output.dim() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "output index shape mismatch at vertex {vertex}: rhs has {}, operator internal output has {}",
                    true_output.dim(),
                    internal_output.dim()
                ),
            ));
        }

        if input_mapping.contains_key(&vertex) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("duplicate mapped vertex {vertex} in linsolve input mapping"),
            ));
        }

        input_mapping.insert(
            vertex,
            IndexMapping {
                true_index: true_input.clone(),
                internal_index: internal_input.clone(),
            },
        );
        output_mapping.insert(
            vertex,
            IndexMapping {
                true_index: true_output.clone(),
                internal_index: internal_output.clone(),
            },
        );
    }

    Ok(Some((input_mapping, output_mapping)))
}

/// Create a tree tensor network from an array of tensors.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_new(
    tensors: *const *const t4a_tensor,
    n_tensors: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        if n_tensors == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "n_tensors must be greater than zero",
            ));
        }
        if tensors.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "tensors is null"));
        }
        checked_ptr_span::<*const t4a_tensor>(n_tensors, "tensors")?;

        let mut tensor_vec = Vec::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let tensor_ptr = unsafe { *tensors.add(i) };
            if tensor_ptr.is_null() {
                return Err(capi_error(
                    T4A_NULL_POINTER,
                    format!("tensors[{i}] is null"),
                ));
            }
            tensor_vec.push(unsafe { &*tensor_ptr }.inner().clone());
        }

        let node_names: Vec<usize> = (0..n_tensors).collect();
        let treetn = InternalTreeTN::from_tensors(tensor_vec, node_names)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(treetn))
    })
}

/// Get the number of vertices in the tree tensor network.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_num_vertices(
    treetn: *const t4a_treetn,
    out_n: *mut libc::size_t,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree(treetn)?;
        if out_n.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_n is null"));
        }
        unsafe { *out_n = tn.inner().node_count() };
        Ok(())
    })
}

/// Get the tensor at a specific vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_tensor(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    let tn = match require_tree(treetn) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let node_idx = tn.inner().node_index(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        let tensor = tn.inner().tensor(node_idx).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} has no tensor"),
            )
        })?;
        Ok(t4a_tensor::new(tensor.clone()))
    })
}

/// Replace the tensor at a specific vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_set_tensor(
    treetn: *mut t4a_treetn,
    vertex: libc::size_t,
    tensor: *const t4a_tensor,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;
        if tensor.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "tensor is null"));
        }
        let node_idx = tn.inner().node_index(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        tn.inner_mut()
            .replace_tensor(node_idx, unsafe { &*tensor }.inner().clone())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(())
    })
}

/// Get the neighbors of a vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_neighbors(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    buf: *mut libc::size_t,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree(treetn)?;
        require_node(tn.inner(), vertex)?;
        let neighbors: Vec<_> = tn.inner().site_index_network().neighbors(&vertex).collect();
        query_then_fill_copy(&neighbors, buf, buf_len, out_len, "t4a_treetn_neighbors")
    })
}

/// Get the canonical region vertices, sorted ascending.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_canonical_region(
    treetn: *const t4a_treetn,
    buf: *mut libc::size_t,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree(treetn)?;
        let mut vertices: Vec<_> = tn.inner().canonical_region().iter().cloned().collect();
        vertices.sort_unstable();
        query_then_fill_copy(&vertices, buf, buf_len, out_len, "canonical_region")
    })
}

/// Get the site indices attached to a vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_site_indices(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    buf: *mut *mut t4a_index,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree(treetn)?;
        let node_idx = tn.inner().node_index(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        let tensor = tn.inner().tensor(node_idx).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} has no tensor"),
            )
        })?;
        let site_space = tn.inner().site_space(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        let ordered_indices: Vec<_> = tensor
            .indices()
            .iter()
            .filter(|index| site_space.contains(*index))
            .cloned()
            .collect();

        if out_len.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_len is null"));
        }
        unsafe { *out_len = ordered_indices.len() };

        if buf.is_null() {
            return Ok(());
        }
        checked_ptr_span::<*mut t4a_index>(ordered_indices.len(), "t4a_treetn_site_indices.buf")?;
        if buf_len < ordered_indices.len() {
            return Err(capi_error(
                T4A_BUFFER_TOO_SMALL,
                format!(
                    "t4a_treetn_site_indices: buffer too small, need {}, got {}",
                    ordered_indices.len(),
                    buf_len
                ),
            ));
        }

        for (i, index) in ordered_indices.iter().enumerate() {
            unsafe { *buf.add(i) = Box::into_raw(Box::new(t4a_index::new(index.clone()))) };
        }
        Ok(())
    })
}

/// Get the bond index on the edge between two vertices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_linkind(
    treetn: *const t4a_treetn,
    v1: libc::size_t,
    v2: libc::size_t,
    out: *mut *mut t4a_index,
) -> t4a_status_code {
    let tn = match require_tree(treetn) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let edge = tn.inner().edge_between(&v1, &v2).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertices {v1} and {v2} are not adjacent"),
            )
        })?;
        let index = tn.inner().bond_index(edge).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("missing bond index between {v1} and {v2}"),
            )
        })?;
        Ok(t4a_index::new(index.clone()))
    })
}

/// Orthogonalize the tree tensor network to a vertex using the requested form.
///
/// When `force == 0`, this uses smart canonicalization:
/// - If the network is already canonical at `vertex` with `form`, the call is a no-op.
/// - If the network is already canonicalized with a different form, the call returns
///
///   `T4A_INVALID_ARGUMENT`. Pass a nonzero `force` to re-canonicalize with a different form.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_orthogonalize(
    treetn: *mut t4a_treetn,
    vertex: libc::size_t,
    form: t4a_canonical_form,
    force: libc::c_int,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;
        require_node(tn.inner(), vertex)?;
        let mut options = CanonicalizationOptions::new().with_form(form.into());
        if force != 0 {
            options = options.force();
        }
        let mut canonicalized = tn.inner().clone();
        canonicalized
            .canonicalize_mut(std::iter::once(vertex), options)
            .map_err(|err| {
                capi_error(
                    T4A_INVALID_ARGUMENT,
                    orthogonalize_error_message(force, err.source),
                )
            })?;
        *tn.inner_mut() = canonicalized;
        Ok(())
    })
}

/// Truncate the tree tensor network bond dimensions using SVD-based truncation.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_truncate(
    treetn: *mut t4a_treetn,
    policy: *const t4a_svd_truncation_policy,
    maxdim: libc::size_t,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;

        let mut options = TruncationOptions::new();
        if let Some(policy) = resolve_svd_policy(policy) {
            options = options.with_svd_policy(policy);
        }
        if maxdim > 0 {
            options = options.with_max_bond_dim(maxdim);
        }

        let center =
            tn.inner().node_names().into_iter().min().ok_or_else(|| {
                capi_error(T4A_INVALID_ARGUMENT, "cannot truncate an empty TreeTN")
            })?;
        tn.inner_mut()
            .truncate_mut(std::iter::once(center), options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(())
    })
}

/// Fuse connected current-node groups into the requested target topology.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_fuse_to(
    treetn: *const t4a_treetn,
    target_vertices: *const libc::size_t,
    n_target_vertices: libc::size_t,
    target_site_indices: *const *const t4a_index,
    target_site_indices_len: *const libc::size_t,
    target_edge_sources: *const libc::size_t,
    target_edge_targets: *const libc::size_t,
    n_target_edges: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        let tn = require_tree(treetn)?;
        let target = build_target_network(
            target_vertices,
            n_target_vertices,
            target_site_indices,
            target_site_indices_len,
            target_edge_sources,
            target_edge_targets,
            n_target_edges,
            "target",
        )?;
        let result = tn
            .inner()
            .fuse_to(&target)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Split current nodes to match the requested target topology.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_split_to(
    treetn: *const t4a_treetn,
    target_vertices: *const libc::size_t,
    n_target_vertices: libc::size_t,
    target_site_indices: *const *const t4a_index,
    target_site_indices_len: *const libc::size_t,
    target_edge_sources: *const libc::size_t,
    target_edge_targets: *const libc::size_t,
    n_target_edges: libc::size_t,
    policy: *const t4a_svd_truncation_policy,
    maxdim: libc::size_t,
    final_sweep: libc::c_int,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        let tn = require_tree(treetn)?;
        let target = build_target_network(
            target_vertices,
            n_target_vertices,
            target_site_indices,
            target_site_indices_len,
            target_edge_sources,
            target_edge_targets,
            n_target_edges,
            "target",
        )?;
        let options = build_split_options(policy, maxdim, final_sweep);
        let result = tn
            .inner()
            .split_to(&target, &options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Reassign site indices to target vertices using scheduled swap transport.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_swap_site_indices(
    treetn: *const t4a_treetn,
    assignment_site_indices: *const *const t4a_index,
    assignment_target_vertices: *const libc::size_t,
    n_assignments: libc::size_t,
    maxdim: libc::size_t,
    rtol: libc::c_double,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        let tn = require_tree(treetn)?;
        let target_assignment = collect_target_assignment(
            assignment_site_indices,
            assignment_target_vertices,
            n_assignments,
            "target_assignment",
        )?;
        let options = build_swap_options(maxdim, rtol);
        let mut result = tn.inner().clone();
        result
            .swap_site_indices(&target_assignment, &options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Restructure a TreeTN using split, swap, and optional final truncation phases.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_restructure_to(
    treetn: *const t4a_treetn,
    target_vertices: *const libc::size_t,
    n_target_vertices: libc::size_t,
    target_site_indices: *const *const t4a_index,
    target_site_indices_len: *const libc::size_t,
    target_edge_sources: *const libc::size_t,
    target_edge_targets: *const libc::size_t,
    n_target_edges: libc::size_t,
    split_policy: *const t4a_svd_truncation_policy,
    split_maxdim: libc::size_t,
    split_final_sweep: libc::c_int,
    swap_maxdim: libc::size_t,
    swap_rtol: libc::c_double,
    final_policy: *const t4a_svd_truncation_policy,
    final_maxdim: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        let tn = require_tree(treetn)?;
        let target = build_target_network(
            target_vertices,
            n_target_vertices,
            target_site_indices,
            target_site_indices_len,
            target_edge_sources,
            target_edge_targets,
            n_target_edges,
            "target",
        )?;
        let mut options = RestructureOptions::new()
            .with_split(build_split_options(
                split_policy,
                split_maxdim,
                split_final_sweep,
            ))
            .with_swap(build_swap_options(swap_maxdim, swap_rtol));
        if let Some(final_truncation) = build_final_truncation(final_policy, final_maxdim) {
            options = options.with_final_truncation(final_truncation);
        }
        let result = tn
            .inner()
            .restructure_to(&target, &options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Create a reusable TreeTN evaluator from explicit index handles.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluator_new(
    treetn: *const t4a_treetn,
    indices: *const *const t4a_index,
    n_indices: libc::size_t,
    out: *mut *mut t4a_treetn_evaluator,
) -> t4a_status_code {
    run_catching(out, || {
        let tn = require_tree(treetn)?;
        if n_indices == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluator_new requires n_indices > 0",
            ));
        }
        let indices = collect_indices(indices, n_indices, "indices")?;
        TreeTNCachedEvaluator::new(tn.inner(), &indices, CachedEvaluatorOptions::default())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn_evaluator::new(InternalTreeTNEvaluatorHandle {
            tree: tn.inner().clone(),
            indices,
            center: None,
            scalar_kind: treetn_scalar_kind(tn.inner()),
        }))
    })
}

/// Evaluate one or more points using a reusable TreeTN evaluator.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluator_evaluate(
    evaluator: *mut t4a_treetn_evaluator,
    values_col_major: *const libc::size_t,
    n_points: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> t4a_status_code {
    run_status(|| {
        let evaluator = require_evaluator_mut(evaluator)?;
        if n_points == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluator_evaluate requires n_points > 0",
            ));
        }
        if values_col_major.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "values_col_major is null"));
        }

        let handle = evaluator.inner_mut();
        let n_indices = handle.indices.len();
        let n_values = n_indices.checked_mul(n_points).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluator_evaluate value array size overflowed size_t",
            )
        })?;
        checked_ptr_span::<libc::size_t>(n_values, "values_col_major")?;
        let values_slice = unsafe { std::slice::from_raw_parts(values_col_major, n_values) };
        let shape = [n_indices, n_points];
        let values = ColMajorArrayRef::new(values_slice, &shape)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        let scalar_kind = handle.scalar_kind;
        let mut cached = TreeTNCachedEvaluator::new(
            &handle.tree,
            &handle.indices,
            CachedEvaluatorOptions {
                center: handle.center,
                ..Default::default()
            },
        )
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let results = cached
            .evaluate_batched(values)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let center = cached.center().copied();
        drop(cached);
        handle.center = center;

        write_evaluation_results(&results, scalar_kind, out_re, out_im)
    })
}

/// Evaluate a TreeTN at one or more points using explicit index handles.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluate(
    treetn: *const t4a_treetn,
    indices: *const *const t4a_index,
    n_indices: libc::size_t,
    values_col_major: *const libc::size_t,
    n_points: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree(treetn)?;
        if n_indices == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluate requires n_indices > 0",
            ));
        }
        if n_points == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluate requires n_points > 0",
            ));
        }
        if values_col_major.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "values_col_major is null"));
        }

        let indices = collect_indices(indices, n_indices, "indices")?;
        let n_values = n_indices.checked_mul(n_points).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluate value array size overflowed size_t",
            )
        })?;
        checked_ptr_span::<libc::size_t>(n_values, "values_col_major")?;
        let values_slice = unsafe { std::slice::from_raw_parts(values_col_major, n_values) };
        let shape = [n_indices, n_points];
        let values = ColMajorArrayRef::new(values_slice, &shape)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let scalar_kind = treetn_scalar_kind(tn.inner());
        let mut evaluator =
            TreeTNCachedEvaluator::new(tn.inner(), &indices, CachedEvaluatorOptions::default())
                .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let results = evaluator
            .evaluate_batched(values)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        write_evaluation_results(&results, scalar_kind, out_re, out_im)
    })
}

/// Compute the inner product of two tree tensor networks.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_inner(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> t4a_status_code {
    run_status(|| {
        let tn_a = require_tree(a)?;
        let tn_b = require_tree(b)?;
        if out_re.is_null() || out_im.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_re or out_im is null"));
        }
        let z: Complex64 = tn_a
            .inner()
            .inner(tn_b.inner())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?
            .into();
        unsafe {
            *out_re = z.re;
            *out_im = z.im;
        }
        Ok(())
    })
}

/// Compute the norm of the tree tensor network.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_norm(
    treetn: *mut t4a_treetn,
    out_norm: *mut libc::c_double,
) -> t4a_status_code {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;
        if out_norm.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_norm is null"));
        }
        unsafe {
            *out_norm = tn
                .inner_mut()
                .norm()
                .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        }
        Ok(())
    })
}

/// Scale a tree tensor network by a complex scalar.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_scale(
    treetn: *const t4a_treetn,
    re: libc::c_double,
    im: libc::c_double,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        let tn = require_tree(treetn)?;
        let scalar = if im == 0.0 {
            AnyScalar::new_real(re)
        } else {
            AnyScalar::new_complex(re, im)
        };
        let result = tn
            .inner()
            .scale(scalar)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Add two tree tensor networks, optionally truncating the result.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_add(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    policy: *const t4a_svd_truncation_policy,
    maxdim: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        let tn_a = require_tree(a)?;
        let tn_b = require_tree(b)?;

        let mut result = tn_a
            .inner()
            .add(tn_b.inner())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        let mut options = TruncationOptions::new();
        if let Some(policy) = resolve_svd_policy(policy) {
            options = options.with_svd_policy(policy);
        }
        if maxdim > 0 {
            options = options.with_max_bond_dim(maxdim);
        }

        if options.svd_policy().is_some() || options.max_bond_dim().is_some() {
            let center = result.node_names().into_iter().min().ok_or_else(|| {
                capi_error(T4A_INVALID_ARGUMENT, "cannot truncate an empty TreeTN")
            })?;
            result
                .truncate_mut(std::iter::once(center), options)
                .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        }

        Ok(t4a_treetn::new(result))
    })
}

/// Contract two tree tensor networks with the requested method.
///
/// `t4a_contract_method::Naive` is a dense/reference method that materializes
/// both TreeTNs and the contracted result as full tensors. Pass a nonzero
/// `max_dense_elements` to opt into this path for small reference cases; each
/// dense input and output tensor must fit under that limit.
///
/// For `t4a_contract_method::Fit`, `nfullsweeps == 0` means "use the backend
/// default", which currently resolves to one variational sweep.
/// For `t4a_contract_method::Src`, `maxdim` must be nonzero and selects the
/// fixed-rank Gaussian-sketch path with default SRC controls. Adaptive SRC
/// controls are currently available through the Rust API only.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_contract(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    method: t4a_contract_method,
    policy: *const t4a_svd_truncation_policy,
    maxdim: libc::size_t,
    nfullsweeps: libc::size_t,
    convergence_tol: libc::c_double,
    factorize_alg: t4a_factorize_alg,
    qr_rtol: libc::c_double,
    max_dense_elements: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let tn_a = match require_tree(a) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    let tn_b = match require_tree(b) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let center =
            tn_a.inner().node_names().into_iter().min().ok_or_else(|| {
                capi_error(T4A_INVALID_ARGUMENT, "cannot contract an empty TreeTN")
            })?;

        let rust_method: ContractionMethod = method.into();
        let fit_nfullsweeps = resolve_fit_nfullsweeps(nfullsweeps);
        let mut options = ContractionOptions::new(rust_method)
            .with_nfullsweeps(fit_nfullsweeps)
            .with_factorize_alg(factorize_alg.into());
        match factorize_alg {
            t4a_factorize_alg::SVD => {
                if qr_rtol != 0.0 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "qr_rtol is only supported when factorize_alg=QR",
                    ));
                }
                if let Some(policy) = resolve_svd_policy(policy) {
                    options = options.with_svd_policy(policy);
                }
            }
            t4a_factorize_alg::QR => {
                if !policy.is_null() {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "policy is only supported when factorize_alg=SVD",
                    ));
                }
                if let Some(rtol) = resolve_qr_rtol(qr_rtol) {
                    options = options.with_qr_rtol(rtol);
                }
            }
            t4a_factorize_alg::LU | t4a_factorize_alg::CI => {
                if !policy.is_null() {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "policy is only supported when factorize_alg=SVD",
                    ));
                }
                if qr_rtol != 0.0 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "qr_rtol is only supported when factorize_alg=QR",
                    ));
                }
            }
        }
        if maxdim > 0 {
            options = options.with_max_bond_dim(maxdim);
        }
        if let Some(tol) = resolve_convergence_tol(convergence_tol) {
            options = options.with_convergence_tol(tol);
        }
        if max_dense_elements > 0 {
            options = options.with_dense_reference_limit(max_dense_elements);
        }

        let result = contraction::contract(tn_a.inner(), tn_b.inner(), &center, options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Partially contract two tree tensor networks using explicit site-index pairs.
///
/// `contract_*` pairs are summed over and removed from the result.
/// `diagonal_*` pairs are linked by diagonal/copy structure while preserving
/// the left-hand index as an external result leg. `output_order`, when nonempty,
/// lists the surviving external indices in the requested result order.
///
/// `t4a_contract_method::Naive` and mismatched-topology fallback paths are
/// dense/reference operations. Pass a nonzero `max_dense_elements` to opt into
/// those paths for small reference cases; each dense input and output tensor
/// must fit under that limit.
///
/// For `t4a_contract_method::Fit`, `nfullsweeps == 0` means "use the backend
/// default", which currently resolves to one variational sweep.
/// For `t4a_contract_method::Src`, `maxdim` must be nonzero and selects the
/// fixed-rank Gaussian-sketch path with default SRC controls. Adaptive SRC
/// controls are currently available through the Rust API only.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn t4a_treetn_partial_contract(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    n_contract_pairs: libc::size_t,
    contract_left: *const *const t4a_index,
    contract_right: *const *const t4a_index,
    n_diagonal_pairs: libc::size_t,
    diagonal_left: *const *const t4a_index,
    diagonal_right: *const *const t4a_index,
    n_output_order: libc::size_t,
    output_order: *const *const t4a_index,
    center: libc::size_t,
    method: t4a_contract_method,
    policy: *const t4a_svd_truncation_policy,
    maxdim: libc::size_t,
    nfullsweeps: libc::size_t,
    convergence_tol: libc::c_double,
    factorize_alg: t4a_factorize_alg,
    qr_rtol: libc::c_double,
    max_dense_elements: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    run_catching(out, || {
        let tn_a = require_tree(a)?;
        let tn_b = require_tree(b)?;
        let contract_pairs = collect_index_pairs(
            contract_left,
            contract_right,
            n_contract_pairs,
            "contract_pairs",
        )?;
        let diagonal_pairs = collect_index_pairs(
            diagonal_left,
            diagonal_right,
            n_diagonal_pairs,
            "diagonal_pairs",
        )?;
        let output_order = if n_output_order == 0 {
            None
        } else {
            Some(collect_indices(
                output_order,
                n_output_order,
                "output_order",
            )?)
        };

        let rust_method: ContractionMethod = method.into();
        let fit_nfullsweeps = resolve_fit_nfullsweeps(nfullsweeps);
        let mut options = ContractionOptions::new(rust_method)
            .with_nfullsweeps(fit_nfullsweeps)
            .with_factorize_alg(factorize_alg.into());
        match factorize_alg {
            t4a_factorize_alg::SVD => {
                if qr_rtol != 0.0 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "qr_rtol is only supported when factorize_alg=QR",
                    ));
                }
                if let Some(policy) = resolve_svd_policy(policy) {
                    options = options.with_svd_policy(policy);
                }
            }
            t4a_factorize_alg::QR => {
                if !policy.is_null() {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "policy is only supported when factorize_alg=SVD",
                    ));
                }
                if let Some(rtol) = resolve_qr_rtol(qr_rtol) {
                    options = options.with_qr_rtol(rtol);
                }
            }
            t4a_factorize_alg::LU | t4a_factorize_alg::CI => {
                if !policy.is_null() {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "policy is only supported when factorize_alg=SVD",
                    ));
                }
                if qr_rtol != 0.0 {
                    return Err(capi_error(
                        T4A_INVALID_ARGUMENT,
                        "qr_rtol is only supported when factorize_alg=QR",
                    ));
                }
            }
        }
        if maxdim > 0 {
            options = options.with_max_bond_dim(maxdim);
        }
        if let Some(tol) = resolve_convergence_tol(convergence_tol) {
            options = options.with_convergence_tol(tol);
        }
        if max_dense_elements > 0 {
            options = options
                .with_dense_reference_limit(max_dense_elements)
                .with_mismatched_topology_dense_limit(max_dense_elements);
        }

        let spec = PartialContractionSpec {
            contract_pairs,
            diagonal_pairs,
            output_order,
        };
        let result = partial_contract(tn_a.inner(), tn_b.inner(), &spec, &center, options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, format!("{err:#}")))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Apply a chain-compatible operator TreeTN to a chain state TreeTN.
///
/// The operator must use dense node names `0..m-1`. `mapped_positions[i]`
/// specifies which state-chain node the operator node `i` acts on. Each mapped
/// operator node must have exactly two site indices, and the corresponding
/// `internal_input_indices[i]` and `internal_output_indices[i]` must point to
/// those indices. Unmapped state nodes are filled with identities by
/// `apply_linear_operator`.
///
/// For `t4a_contract_method::Fit`, `nfullsweeps == 0` means "use the backend
/// default", which currently resolves to one variational sweep.
/// `t4a_contract_method::Naive` uses the dedicated local-exact apply path here,
/// not the generic full-dense TreeTN contraction used by `t4a_treetn_contract`.
/// For `t4a_contract_method::Src`, `maxdim` must be nonzero and selects the
/// fixed-rank Gaussian-sketch path with default SRC controls.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_apply_operator_chain(
    operator: *const t4a_treetn,
    state: *const t4a_treetn,
    mapped_positions: *const libc::size_t,
    n_mapped_positions: libc::size_t,
    internal_input_indices: *const *const t4a_index,
    internal_output_indices: *const *const t4a_index,
    true_output_indices: *const *const t4a_index,
    method: t4a_contract_method,
    policy: *const t4a_svd_truncation_policy,
    maxdim: libc::size_t,
    nfullsweeps: libc::size_t,
    convergence_tol: libc::c_double,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let op = match require_tree(operator) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    let state = match require_tree(state) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let mapped_positions =
            collect_positions(mapped_positions, n_mapped_positions, "mapped_positions")?;
        let internal_inputs = collect_indices(
            internal_input_indices,
            n_mapped_positions,
            "internal_input_indices",
        )?;
        let internal_outputs = collect_indices(
            internal_output_indices,
            n_mapped_positions,
            "internal_output_indices",
        )?;
        let true_outputs = collect_indices(
            true_output_indices,
            n_mapped_positions,
            "true_output_indices",
        )?;

        let state_true_inputs = collect_single_site_indices(state.inner(), "state")?;
        validate_mapped_positions(
            &mapped_positions,
            state_true_inputs.len(),
            op.inner().node_count(),
        )?;
        let linear_operator = build_chain_linear_operator(
            op.inner(),
            &state_true_inputs,
            &mapped_positions,
            &internal_inputs,
            &internal_outputs,
            &true_outputs,
        )?;

        let fit_nfullsweeps = resolve_fit_nfullsweeps(nfullsweeps);
        let result = apply_linear_operator(
            &linear_operator,
            state.inner(),
            ApplyOptions {
                method: method.into(),
                max_bond_dim: (maxdim > 0).then_some(maxdim),
                svd_policy: resolve_svd_policy(policy),
                qr_rtol: None,
                nfullsweeps: fit_nfullsweeps,
                convergence_tol: resolve_convergence_tol(convergence_tol),
                src_options: Default::default(),
            },
        )
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Solve `(a0 + a1 * operator) * x = rhs` for `x` as a TreeTN.
///
/// The solver returns only the solution TreeTN. The Rust-side `SquareLinsolveResult`
/// also tracks sweep metadata, but that is not yet exposed through the C ABI.
///
/// `mapped_vertices` and the four index arrays provide optional per-vertex
/// index mappings when the operator uses internal site indices distinct from
/// the `init` / `rhs` site indices. Pass `n_mapped_vertices == 0` to disable
/// mappings.
///
/// Current limitation: the sweep-based backend still assumes that `init` and
/// `rhs` share the same true site-index set. The mapping arrays bridge the
/// operator's internal indices to those true indices, but they do not yet
/// support solving between distinct `init` and `rhs` true index spaces.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_linsolve(
    operator: *const t4a_treetn,
    rhs: *const t4a_treetn,
    init: *const t4a_treetn,
    center_vertex: libc::size_t,
    mapped_vertices: *const libc::size_t,
    n_mapped_vertices: libc::size_t,
    true_input_indices: *const *const t4a_index,
    internal_input_indices: *const *const t4a_index,
    true_output_indices: *const *const t4a_index,
    internal_output_indices: *const *const t4a_index,
    policy: *const t4a_svd_truncation_policy,
    maxdim: libc::size_t,
    nfullsweeps: libc::size_t,
    gmres_tol: libc::c_double,
    gmres_max_restarts: libc::size_t,
    gmres_restart_dim: libc::size_t,
    a0: libc::c_double,
    a1: libc::c_double,
    convergence_tol: libc::c_double,
    out: *mut *mut t4a_treetn,
) -> t4a_status_code {
    let operator = match require_tree(operator) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    let rhs = match require_tree(rhs) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    let init = match require_tree(init) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        require_node(init.inner(), center_vertex)?;

        if !gmres_tol.is_finite() || gmres_tol <= 0.0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("gmres_tol must be finite and > 0, got {gmres_tol}"),
            ));
        }
        if gmres_max_restarts == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "gmres_max_restarts must be >= 1",
            ));
        }
        if gmres_restart_dim == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "gmres_restart_dim must be >= 1",
            ));
        }
        if !a0.is_finite() || !a1.is_finite() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("a0 and a1 must be finite, got a0={a0}, a1={a1}"),
            ));
        }
        if !(convergence_tol == 0.0 || (convergence_tol.is_finite() && convergence_tol > 0.0)) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "convergence_tol must be 0.0 or a finite positive value, got {convergence_tol}"
                ),
            ));
        }

        let mappings = build_linsolve_index_mappings(
            operator.inner(),
            rhs.inner(),
            init.inner(),
            mapped_vertices,
            n_mapped_vertices,
            true_input_indices,
            internal_input_indices,
            true_output_indices,
            internal_output_indices,
        )?;

        let mut options = LinsolveOptions::new(nfullsweeps)
            .with_truncation(TruncationOptions::new())
            .with_gmres_tol(gmres_tol)
            .with_gmres_max_restarts(gmres_max_restarts)
            .with_gmres_restart_dim(gmres_restart_dim)
            .with_coefficients(a0, a1);
        if let Some(policy) = resolve_svd_policy(policy) {
            options = options.with_svd_policy(policy);
        }
        if maxdim > 0 {
            options = options.with_max_bond_dim(maxdim);
        }
        if let Some(tol) = resolve_convergence_tol(convergence_tol) {
            options = options.with_convergence_tol(tol);
        }

        let (input_mapping, output_mapping) = match mappings {
            Some((input_mapping, output_mapping)) => (Some(input_mapping), Some(output_mapping)),
            None => (None, None),
        };

        let result = square_linsolve(
            operator.inner(),
            rhs.inner(),
            init.inner().clone(),
            &center_vertex,
            options,
            input_mapping,
            output_mapping,
        )
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result.solution))
    })
}

/// Contract all bonds and materialize the TreeTN as a dense tensor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_to_dense(
    treetn: *const t4a_treetn,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    let tn = match require_tree(treetn) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let tensor = tn
            .inner()
            .to_dense()
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(tensor))
    })
}

#[cfg(test)]
mod tests;
