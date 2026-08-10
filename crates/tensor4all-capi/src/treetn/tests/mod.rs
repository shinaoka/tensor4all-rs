use super::*;
use crate::index::{t4a_index_dim, t4a_index_new, t4a_index_release};
use crate::tensor::{
    t4a_tensor_copy_dense_f64, t4a_tensor_new_dense_c64, t4a_tensor_new_dense_f64, t4a_tensor_rank,
    t4a_tensor_release,
};
use crate::types::{t4a_singular_value_measure, t4a_threshold_scale, t4a_truncation_rule};

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

fn new_index(dim: usize) -> *mut t4a_index {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_index_new(dim, std::ptr::null(), 0, &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn new_tensor(indices: &[*const t4a_index], data: &[f64]) -> *mut t4a_tensor {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            indices.len(),
            indices.as_ptr(),
            data.as_ptr(),
            data.len(),
            &mut out
        ),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn read_dense_f64_tensor(tensor: *const t4a_tensor) -> Vec<f64> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_copy_dense_f64(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut data = vec![0.0; len];
    assert_eq!(
        t4a_tensor_copy_dense_f64(tensor, data.as_mut_ptr(), data.len(), &mut len),
        T4A_SUCCESS
    );
    data
}

fn new_treetn(tensors: &[*const t4a_tensor]) -> *mut t4a_treetn {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_new(tensors.as_ptr(), tensors.len(), &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn read_dense_f64_treetn(treetn: *const t4a_treetn) -> Vec<f64> {
    let mut dense = std::ptr::null_mut();
    assert_eq!(t4a_treetn_to_dense(treetn, &mut dense), T4A_SUCCESS);
    let values = read_dense_f64_tensor(dense);
    t4a_tensor_release(dense);
    values
}

fn read_siteinds(tt: *const t4a_treetn, vertex: usize) -> Vec<*mut t4a_index> {
    let mut len = 0usize;
    assert_eq!(
        t4a_treetn_siteinds(tt, vertex, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut indices = vec![std::ptr::null_mut(); len];
    assert_eq!(
        t4a_treetn_siteinds(tt, vertex, indices.as_mut_ptr(), indices.len(), &mut len),
        T4A_SUCCESS
    );
    indices
}

fn read_canonical_region(tt: *const t4a_treetn) -> Vec<usize> {
    let mut len = 0usize;
    assert_eq!(
        t4a_treetn_canonical_region(tt, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut vertices = vec![0usize; len];
    assert_eq!(
        t4a_treetn_canonical_region(tt, vertices.as_mut_ptr(), vertices.len(), &mut len),
        T4A_SUCCESS
    );
    vertices
}

fn index_dim(index: *const t4a_index) -> usize {
    let mut dim = 0usize;
    assert_eq!(t4a_index_dim(index, &mut dim), T4A_SUCCESS);
    dim
}

fn evaluate_real_on_full_grid(tt: *const t4a_treetn, indices: &[*const t4a_index]) -> Vec<f64> {
    let dims: Vec<usize> = indices.iter().map(|&index| index_dim(index)).collect();
    let n_points: usize = dims.iter().product();
    let n_indices = indices.len();
    let mut values_col_major = vec![0usize; n_indices * n_points];
    let mut coords = vec![0usize; n_indices];
    for point in 0..n_points {
        let mut linear = point;
        for axis in (0..n_indices).rev() {
            coords[axis] = linear % dims[axis];
            linear /= dims[axis];
        }
        for axis in 0..n_indices {
            values_col_major[axis + point * n_indices] = coords[axis];
        }
    }

    let mut out_re = vec![0.0; n_points];
    assert_eq!(
        t4a_treetn_evaluate(
            tt,
            indices.as_ptr(),
            n_indices,
            values_col_major.as_ptr(),
            n_points,
            out_re.as_mut_ptr(),
            std::ptr::null_mut()
        ),
        T4A_SUCCESS
    );
    out_re
}

fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (slot, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (a - e).abs();
        assert!(
            err <= tol,
            "value mismatch at slot {slot}: actual={a}, expected={e}, err={err}, tol={tol}"
        );
    }
}

fn relative_policy(threshold: f64) -> t4a_svd_truncation_policy {
    t4a_svd_truncation_policy {
        threshold,
        scale: t4a_threshold_scale::Relative,
        measure: t4a_singular_value_measure::Value,
        rule: t4a_truncation_rule::PerValue,
    }
}

fn squared_tail_policy(threshold: f64) -> t4a_svd_truncation_policy {
    t4a_svd_truncation_policy {
        threshold,
        scale: t4a_threshold_scale::Relative,
        measure: t4a_singular_value_measure::SquaredValue,
        rule: t4a_truncation_rule::DiscardedTailSum,
    }
}

type OwnedTreeTN = (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>);

fn make_two_site_treetn() -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
    let s0 = new_index(2);
    let bond = new_index(2);
    let s1 = new_index(2);
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );

    // t0(s0, bond) = [[1, 0], [0, 1]]
    let t0 = new_tensor(
        &[s0 as *const t4a_index, bond as *const t4a_index],
        &[1.0, 0.0, 0.0, 1.0],
    );
    // t1(bond, s1) = [[1, 3], [2, 4]]
    let t1 = new_tensor(
        &[bond_clone as *const t4a_index, s1 as *const t4a_index],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let tt = new_treetn(&[t0 as *const t4a_tensor, t1 as *const t4a_tensor]);
    (tt, vec![t0, t1], vec![s0, bond, bond_clone, s1])
}

fn make_two_site_identity_operator_with_shared_inputs(
    input0: *const t4a_index,
    input1: *const t4a_index,
) -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
    let out0 = new_index(index_dim(input0));
    let out1 = new_index(index_dim(input1));
    let bond = new_index(1);
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );

    let t0 = new_tensor(
        &[out0 as *const t4a_index, input0, bond as *const t4a_index],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let t1 = new_tensor(
        &[
            bond_clone as *const t4a_index,
            out1 as *const t4a_index,
            input1,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let tt = new_treetn(&[t0 as *const t4a_tensor, t1 as *const t4a_tensor]);
    (tt, vec![t0, t1], vec![out0, out1, bond, bond_clone])
}

fn make_two_site_identity_operator_with_internal_indices() -> (
    *mut t4a_treetn,
    Vec<*mut t4a_tensor>,
    Vec<*mut t4a_index>,
    [*mut t4a_index; 2],
    [*mut t4a_index; 2],
) {
    let internal_inputs = [new_index(2), new_index(2)];
    let internal_outputs = [new_index(2), new_index(2)];
    let bond = new_index(1);
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );

    let t0 = new_tensor(
        &[
            internal_outputs[0] as *const t4a_index,
            internal_inputs[0] as *const t4a_index,
            bond as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let t1 = new_tensor(
        &[
            bond_clone as *const t4a_index,
            internal_outputs[1] as *const t4a_index,
            internal_inputs[1] as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let tt = new_treetn(&[t0 as *const t4a_tensor, t1 as *const t4a_tensor]);
    let indices = vec![
        internal_inputs[0],
        internal_inputs[1],
        internal_outputs[0],
        internal_outputs[1],
        bond,
        bond_clone,
    ];
    (tt, vec![t0, t1], indices, internal_inputs, internal_outputs)
}

fn make_two_node_groups_of_two_treetn(
) -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
    let x0 = new_index(2);
    let x1 = new_index(2);
    let y0 = new_index(2);
    let y1 = new_index(2);
    let bond = new_index(2);
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );

    let left = new_tensor(
        &[
            x0 as *const t4a_index,
            x1 as *const t4a_index,
            bond as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    );
    let right = new_tensor(
        &[
            bond_clone as *const t4a_index,
            y0 as *const t4a_index,
            y1 as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    );
    let tt = new_treetn(&[left as *const t4a_tensor, right as *const t4a_tensor]);
    (
        tt,
        vec![left, right],
        vec![x0, x1, y0, y1, bond, bond_clone],
    )
}

struct FlatTargetNetwork {
    vertices: Vec<usize>,
    siteinds: Vec<*const t4a_index>,
    siteind_lens: Vec<usize>,
    edge_sources: Vec<usize>,
    edge_targets: Vec<usize>,
}

fn flat_target_network(
    vertices: &[usize],
    site_groups: &[&[*const t4a_index]],
    edges: &[(usize, usize)],
) -> FlatTargetNetwork {
    assert_eq!(vertices.len(), site_groups.len());
    let mut siteinds = Vec::new();
    let mut siteind_lens = Vec::with_capacity(site_groups.len());
    for group in site_groups {
        siteind_lens.push(group.len());
        siteinds.extend(group.iter().copied());
    }
    let mut edge_sources = Vec::with_capacity(edges.len());
    let mut edge_targets = Vec::with_capacity(edges.len());
    for &(src, dst) in edges {
        edge_sources.push(src);
        edge_targets.push(dst);
    }
    FlatTargetNetwork {
        vertices: vertices.to_vec(),
        siteinds,
        siteind_lens,
        edge_sources,
        edge_targets,
    }
}

struct FlatAssignment {
    siteinds: Vec<*const t4a_index>,
    target_vertices: Vec<usize>,
}

fn flat_assignment(assignments: &[(*const t4a_index, usize)]) -> FlatAssignment {
    let mut siteinds = Vec::with_capacity(assignments.len());
    let mut target_vertices = Vec::with_capacity(assignments.len());
    for &(index, target_vertex) in assignments {
        siteinds.push(index);
        target_vertices.push(target_vertex);
    }
    FlatAssignment {
        siteinds,
        target_vertices,
    }
}

fn make_two_site_contract_operands() -> (OwnedTreeTN, OwnedTreeTN) {
    let in0 = new_index(2);
    let in1 = new_index(2);
    let out0 = new_index(2);
    let out1 = new_index(2);

    let state_bond = new_index(2);
    let mut state_bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(state_bond, &mut state_bond_clone),
        T4A_SUCCESS
    );

    let op_bond = new_index(2);
    let mut op_bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(op_bond, &mut op_bond_clone),
        T4A_SUCCESS
    );

    let mut in0_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(in0, &mut in0_clone),
        T4A_SUCCESS
    );
    let mut in1_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(in1, &mut in1_clone),
        T4A_SUCCESS
    );

    let state_t0 = new_tensor(
        &[in0 as *const t4a_index, state_bond as *const t4a_index],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let state_t1 = new_tensor(
        &[
            state_bond_clone as *const t4a_index,
            in1 as *const t4a_index,
        ],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let state_tt = new_treetn(&[state_t0 as *const t4a_tensor, state_t1 as *const t4a_tensor]);

    let op_t0 = new_tensor(
        &[
            out0 as *const t4a_index,
            in0_clone as *const t4a_index,
            op_bond as *const t4a_index,
        ],
        &[1.0, 0.0, 0.5, -0.5, 0.0, 1.0, 1.5, 0.25],
    );
    let op_t1 = new_tensor(
        &[
            op_bond_clone as *const t4a_index,
            out1 as *const t4a_index,
            in1_clone as *const t4a_index,
        ],
        &[1.0, 2.0, 0.0, 1.0, 2.0, -1.0, 1.0, 0.5],
    );
    let op_tt = new_treetn(&[op_t0 as *const t4a_tensor, op_t1 as *const t4a_tensor]);

    (
        (
            op_tt,
            vec![op_t0, op_t1],
            vec![out0, out1, in0_clone, in1_clone, op_bond, op_bond_clone],
        ),
        (
            state_tt,
            vec![state_t0, state_t1],
            vec![in0, in1, state_bond, state_bond_clone],
        ),
    )
}

fn cleanup(tt: *mut t4a_treetn, tensors: Vec<*mut t4a_tensor>, indices: Vec<*mut t4a_index>) {
    t4a_treetn_release(tt);
    for tensor in tensors {
        t4a_tensor_release(tensor);
    }
    for index in indices {
        t4a_index_release(index);
    }
}

#[test]
fn test_treetn_topology_queries() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut n_vertices = 0usize;
    assert_eq!(t4a_treetn_num_vertices(tt, &mut n_vertices), T4A_SUCCESS);
    assert_eq!(n_vertices, 2);

    let mut neigh_len = 0usize;
    assert_eq!(
        t4a_treetn_neighbors(tt, 0, std::ptr::null_mut(), 0, &mut neigh_len),
        T4A_SUCCESS
    );
    assert_eq!(neigh_len, 1);
    let mut neighbors = vec![0usize; neigh_len];
    assert_eq!(
        t4a_treetn_neighbors(
            tt,
            0,
            neighbors.as_mut_ptr(),
            neighbors.len(),
            &mut neigh_len
        ),
        T4A_SUCCESS
    );
    assert_eq!(neighbors, vec![1]);

    let mut siteind_len = 0usize;
    assert_eq!(
        t4a_treetn_siteinds(tt, 0, std::ptr::null_mut(), 0, &mut siteind_len),
        T4A_SUCCESS
    );
    assert_eq!(siteind_len, 1);
    let mut siteinds = vec![std::ptr::null_mut(); siteind_len];
    assert_eq!(
        t4a_treetn_siteinds(
            tt,
            0,
            siteinds.as_mut_ptr(),
            siteinds.len(),
            &mut siteind_len
        ),
        T4A_SUCCESS
    );
    let mut site_dim = 0usize;
    assert_eq!(
        crate::index::t4a_index_dim(siteinds[0], &mut site_dim),
        T4A_SUCCESS
    );
    assert_eq!(site_dim, 2);
    t4a_index_release(siteinds[0]);

    let mut link = std::ptr::null_mut();
    assert_eq!(t4a_treetn_linkind(tt, 0, 1, &mut link), T4A_SUCCESS);
    let mut link_dim = 0usize;
    assert_eq!(
        crate::index::t4a_index_dim(link, &mut link_dim),
        T4A_SUCCESS
    );
    assert_eq!(link_dim, 2);
    t4a_index_release(link);

    let mut tensor = std::ptr::null_mut();
    assert_eq!(t4a_treetn_tensor(tt, 0, &mut tensor), T4A_SUCCESS);
    let mut rank = 0usize;
    assert_eq!(t4a_tensor_rank(tensor, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 2);
    t4a_tensor_release(tensor);

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_clone_inner_norm_and_assignment() {
    let (tt, tensors, indices) = make_two_site_treetn();

    assert_eq!(t4a_treetn_is_assigned(tt), 1);
    assert_eq!(t4a_treetn_is_assigned(std::ptr::null()), 0);

    let mut clone = std::ptr::null_mut();
    assert_eq!(t4a_treetn_clone(tt, &mut clone), T4A_SUCCESS);
    assert!(!clone.is_null());

    let mut inner_re = 0.0;
    let mut inner_im = 0.0;
    assert_eq!(
        t4a_treetn_inner(tt, clone, &mut inner_re, &mut inner_im),
        T4A_SUCCESS
    );
    assert!((inner_re - 30.0).abs() < 1e-12);
    assert_eq!(inner_im, 0.0);

    let mut norm = 0.0;
    assert_eq!(t4a_treetn_norm(clone, &mut norm), T4A_SUCCESS);
    assert!((norm * norm - inner_re).abs() < 1e-12);

    t4a_treetn_release(clone);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_query_helpers_reject_invalid_requests() {
    let (tt, tensors, indices) = make_two_site_treetn();

    assert_eq!(
        t4a_treetn_num_vertices(tt, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_n is null"));

    let mut neigh_len = 0usize;
    let mut neighbors: [usize; 0] = [];
    assert_eq!(
        t4a_treetn_neighbors(tt, 0, neighbors.as_mut_ptr(), 0, &mut neigh_len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(neigh_len, 1);

    assert_eq!(
        t4a_treetn_neighbors(tt, 2, std::ptr::null_mut(), 0, &mut neigh_len),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("vertex 2 does not exist"));

    let mut siteind_len = 0usize;
    let mut siteinds: [*mut t4a_index; 0] = [];
    assert_eq!(
        t4a_treetn_siteinds(tt, 0, siteinds.as_mut_ptr(), 0, &mut siteind_len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(siteind_len, 1);

    let mut tensor = std::ptr::null_mut();
    assert_eq!(t4a_treetn_tensor(tt, 9, &mut tensor), T4A_INVALID_ARGUMENT);
    assert!(tensor.is_null());

    let mut link = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_linkind(tt, 0, 0, &mut link),
        T4A_INVALID_ARGUMENT
    );
    assert!(link.is_null());

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_creation_and_scalar_api_validation_errors() {
    let mut tt = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_new(std::ptr::null(), 0, &mut tt),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("n_tensors must be greater than zero"));

    assert_eq!(
        t4a_treetn_new(std::ptr::null(), 1, &mut tt),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("tensors is null"));

    let (tt, tensors, indices) = make_two_site_treetn();
    assert_eq!(
        t4a_treetn_set_tensor(tt, 9, tensors[0] as *const t4a_tensor),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("vertex 9 does not exist"));

    assert_eq!(
        t4a_treetn_set_tensor(tt, 0, std::ptr::null()),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("tensor is null"));

    let mut inner_re = 0.0;
    assert_eq!(
        t4a_treetn_inner(tt, tt, &mut inner_re, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_re or out_im is null"));

    assert_eq!(t4a_treetn_norm(tt, std::ptr::null_mut()), T4A_NULL_POINTER);
    assert!(last_error().contains("out_norm is null"));

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_set_tensor_and_to_dense() {
    let (tt, mut tensors, mut indices) = make_two_site_treetn();

    let bond = indices[1];
    let s1 = indices[3];
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );
    let replacement = new_tensor(
        &[bond_clone as *const t4a_index, s1 as *const t4a_index],
        &[10.0, 20.0, 30.0, 40.0],
    );
    assert_eq!(t4a_treetn_set_tensor(tt, 1, replacement), T4A_SUCCESS);
    assert_eq!(read_dense_f64_treetn(tt), vec![10.0, 20.0, 30.0, 40.0]);

    tensors.push(replacement);
    indices.push(bond_clone);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_fuse_to_merges_two_vertices() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let target = flat_target_network(
        &[7],
        &[&[
            indices[0] as *const t4a_index,
            indices[3] as *const t4a_index,
        ]],
        &[],
    );

    let mut fused = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_fuse_to(
            tt,
            target.vertices.as_ptr(),
            target.vertices.len(),
            target.siteinds.as_ptr(),
            target.siteind_lens.as_ptr(),
            target.edge_sources.as_ptr(),
            target.edge_targets.as_ptr(),
            target.edge_sources.len(),
            &mut fused,
        ),
        T4A_SUCCESS
    );

    let dense_expected = read_dense_f64_treetn(tt);
    let dense_actual = read_dense_f64_treetn(fused);
    assert_vec_close(&dense_actual, &dense_expected, 1e-10);
    assert_eq!(read_siteinds(fused, 7).len(), 2);

    t4a_treetn_release(fused);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_split_to_splits_fused_vertex() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let fuse_target = flat_target_network(
        &[11],
        &[&[
            indices[0] as *const t4a_index,
            indices[3] as *const t4a_index,
        ]],
        &[],
    );
    let mut fused = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_fuse_to(
            tt,
            fuse_target.vertices.as_ptr(),
            fuse_target.vertices.len(),
            fuse_target.siteinds.as_ptr(),
            fuse_target.siteind_lens.as_ptr(),
            fuse_target.edge_sources.as_ptr(),
            fuse_target.edge_targets.as_ptr(),
            fuse_target.edge_sources.len(),
            &mut fused,
        ),
        T4A_SUCCESS
    );

    let split_target = flat_target_network(
        &[3, 8],
        &[
            &[indices[0] as *const t4a_index],
            &[indices[3] as *const t4a_index],
        ],
        &[(3, 8)],
    );
    let mut split = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_split_to(
            fused,
            split_target.vertices.as_ptr(),
            split_target.vertices.len(),
            split_target.siteinds.as_ptr(),
            split_target.siteind_lens.as_ptr(),
            split_target.edge_sources.as_ptr(),
            split_target.edge_targets.as_ptr(),
            split_target.edge_sources.len(),
            std::ptr::null(),
            0,
            0,
            &mut split,
        ),
        T4A_SUCCESS
    );

    let dense_expected = read_dense_f64_treetn(tt);
    let dense_actual = read_dense_f64_treetn(split);
    assert_vec_close(&dense_actual, &dense_expected, 1e-10);
    assert_eq!(read_siteinds(split, 3).len(), 1);
    assert_eq!(read_siteinds(split, 8).len(), 1);

    t4a_treetn_release(split);
    t4a_treetn_release(fused);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_swap_site_indices_returns_reordered_copy() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let assignment = flat_assignment(&[
        (indices[0] as *const t4a_index, 1),
        (indices[3] as *const t4a_index, 0),
    ]);

    let mut swapped = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_swap_site_indices(
            tt,
            assignment.siteinds.as_ptr(),
            assignment.target_vertices.as_ptr(),
            assignment.siteinds.len(),
            0,
            0.0,
            &mut swapped,
        ),
        T4A_SUCCESS
    );

    let eval_indices = [
        indices[0] as *const t4a_index,
        indices[3] as *const t4a_index,
    ];
    let dense_expected = evaluate_real_on_full_grid(tt, &eval_indices);
    let dense_actual = evaluate_real_on_full_grid(swapped, &eval_indices);
    assert_vec_close(&dense_actual, &dense_expected, 1e-10);
    assert_eq!(read_siteinds(swapped, 0).len(), 1);
    assert_eq!(read_siteinds(swapped, 1).len(), 1);

    t4a_treetn_release(swapped);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_swap_site_indices_rejects_duplicate_assignment() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let assignment = flat_assignment(&[
        (indices[0] as *const t4a_index, 0),
        (indices[0] as *const t4a_index, 1),
    ]);

    let mut swapped = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_swap_site_indices(
            tt,
            assignment.siteinds.as_ptr(),
            assignment.target_vertices.as_ptr(),
            assignment.siteinds.len(),
            0,
            0.0,
            &mut swapped,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("duplicate site index assignment"));

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_restructure_to_mixed_case() {
    let (tt, tensors, indices) = make_two_node_groups_of_two_treetn();
    let target = flat_target_network(
        &[10, 20, 30],
        &[
            &[indices[0] as *const t4a_index],
            &[
                indices[1] as *const t4a_index,
                indices[2] as *const t4a_index,
            ],
            &[indices[3] as *const t4a_index],
        ],
        &[(10, 20), (20, 30)],
    );

    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_restructure_to(
            tt,
            target.vertices.as_ptr(),
            target.vertices.len(),
            target.siteinds.as_ptr(),
            target.siteind_lens.as_ptr(),
            target.edge_sources.as_ptr(),
            target.edge_targets.as_ptr(),
            target.edge_sources.len(),
            std::ptr::null(),
            0,
            0,
            0,
            0.0,
            std::ptr::null(),
            0,
            &mut result,
        ),
        T4A_SUCCESS
    );

    let eval_indices = [
        indices[0] as *const t4a_index,
        indices[1] as *const t4a_index,
        indices[2] as *const t4a_index,
        indices[3] as *const t4a_index,
    ];
    let dense_expected = evaluate_real_on_full_grid(tt, &eval_indices);
    let dense_actual = evaluate_real_on_full_grid(result, &eval_indices);
    assert_vec_close(&dense_actual, &dense_expected, 1e-10);
    assert_eq!(read_siteinds(result, 10).len(), 1);
    assert_eq!(read_siteinds(result, 20).len(), 2);
    assert_eq!(read_siteinds(result, 30).len(), 1);

    t4a_treetn_release(result);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_fuse_to_rejects_duplicate_target_site_index() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let target = flat_target_network(
        &[5, 9],
        &[
            &[indices[0] as *const t4a_index],
            &[
                indices[0] as *const t4a_index,
                indices[3] as *const t4a_index,
            ],
        ],
        &[(5, 9)],
    );

    let mut fused = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_fuse_to(
            tt,
            target.vertices.as_ptr(),
            target.vertices.len(),
            target.siteinds.as_ptr(),
            target.siteind_lens.as_ptr(),
            target.edge_sources.as_ptr(),
            target.edge_targets.as_ptr(),
            target.edge_sources.len(),
            &mut fused,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("duplicate site index"));

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_orthogonalize_preserves_dense_tensor() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let before = read_dense_f64_treetn(tt);
    assert_eq!(
        t4a_treetn_orthogonalize(tt, 0, t4a_canonical_form::LU, 1),
        T4A_SUCCESS
    );
    let after = read_dense_f64_treetn(tt);
    assert_eq!(before, after);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_canonical_region_query() {
    let (tt, tensors, indices) = make_two_site_treetn();
    assert_eq!(read_canonical_region(tt), Vec::<usize>::new());

    let before = read_dense_f64_treetn(tt);
    assert_eq!(
        t4a_treetn_orthogonalize(tt, 1, t4a_canonical_form::Unitary, 0),
        T4A_SUCCESS
    );
    assert_eq!(read_canonical_region(tt), vec![1]);
    let canonicalized = read_dense_f64_treetn(tt);
    assert_eq!(canonicalized, before);

    assert_eq!(
        t4a_treetn_orthogonalize(tt, 1, t4a_canonical_form::Unitary, 0),
        T4A_SUCCESS
    );
    assert_eq!(read_canonical_region(tt), vec![1]);
    assert_eq!(read_dense_f64_treetn(tt), canonicalized);

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_orthogonalize_force_zero_form_mismatch_reports_c_api_hint() {
    let (tt, tensors, indices) = make_two_site_treetn();
    assert_eq!(
        t4a_treetn_orthogonalize(tt, 1, t4a_canonical_form::Unitary, 0),
        T4A_SUCCESS
    );
    let before = read_dense_f64_treetn(tt);

    assert_eq!(
        t4a_treetn_orthogonalize(tt, 1, t4a_canonical_form::LU, 0),
        T4A_INVALID_ARGUMENT
    );
    let err = last_error();
    assert!(
        err.contains("force=1"),
        "unexpected error message for force=0 form mismatch: {err}"
    );
    assert_eq!(read_dense_f64_treetn(tt), before);

    assert_eq!(
        t4a_treetn_orthogonalize(tt, 1, t4a_canonical_form::LU, 1),
        T4A_SUCCESS
    );
    assert_eq!(read_dense_f64_treetn(tt), before);

    cleanup(tt, tensors, indices);
}

fn make_truncatable_treetn() -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
    let s0 = new_index(2);
    let bond = new_index(4);
    let s1 = new_index(2);
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );

    let t0 = new_tensor(
        &[s0 as *const t4a_index, bond as *const t4a_index],
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    );
    let t1 = new_tensor(
        &[bond_clone as *const t4a_index, s1 as *const t4a_index],
        &[1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0],
    );
    let tt = new_treetn(&[t0 as *const t4a_tensor, t1 as *const t4a_tensor]);
    (tt, vec![t0, t1], vec![s0, bond, bond_clone, s1])
}

fn link_dim(tt: *const t4a_treetn) -> usize {
    let mut link = std::ptr::null_mut();
    assert_eq!(t4a_treetn_linkind(tt, 0, 1, &mut link), T4A_SUCCESS);
    let mut dim = 0usize;
    assert_eq!(crate::index::t4a_index_dim(link, &mut dim), T4A_SUCCESS);
    t4a_index_release(link);
    dim
}

#[test]
fn test_treetn_truncate_with_rtol_and_cutoff() {
    let (tt_rtol, tensors_rtol, indices_rtol) = make_truncatable_treetn();
    assert_eq!(link_dim(tt_rtol), 4);
    let rtol_policy = relative_policy(1e-12);
    assert_eq!(t4a_treetn_truncate(tt_rtol, &rtol_policy, 1), T4A_SUCCESS);
    assert_eq!(link_dim(tt_rtol), 1);
    cleanup(tt_rtol, tensors_rtol, indices_rtol);

    let (tt_cutoff, tensors_cutoff, indices_cutoff) = make_truncatable_treetn();
    assert_eq!(link_dim(tt_cutoff), 4);
    let cutoff_policy = squared_tail_policy(1e-24);
    assert_eq!(
        t4a_treetn_truncate(tt_cutoff, &cutoff_policy, 1),
        T4A_SUCCESS
    );
    assert_eq!(link_dim(tt_cutoff), 1);
    cleanup(tt_cutoff, tensors_cutoff, indices_cutoff);
}

#[test]
fn test_treetn_scale_and_add() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut scaled = std::ptr::null_mut();
    assert_eq!(t4a_treetn_scale(tt, 2.0, 0.0, &mut scaled), T4A_SUCCESS);
    assert_eq!(read_dense_f64_treetn(scaled), vec![2.0, 4.0, 6.0, 8.0]);

    let mut sum = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_add(tt, tt, std::ptr::null(), 0, &mut sum),
        T4A_SUCCESS
    );
    assert_eq!(read_dense_f64_treetn(sum), vec![2.0, 4.0, 6.0, 8.0]);

    t4a_treetn_release(sum);
    t4a_treetn_release(scaled);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_add_with_truncation() {
    let (tt, tensors, indices) = make_truncatable_treetn();
    let expected = read_dense_f64_treetn(tt);
    let policy = relative_policy(1e-12);

    let mut sum = std::ptr::null_mut();
    assert_eq!(t4a_treetn_add(tt, tt, &policy, 1, &mut sum), T4A_SUCCESS);
    assert_eq!(link_dim(sum), 1);

    let dense = read_dense_f64_treetn(sum);
    let doubled_expected: Vec<f64> = expected.iter().map(|value| 2.0 * value).collect();
    assert_vec_close(&dense, &doubled_expected, 1.0e-10);

    t4a_treetn_release(sum);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_evaluate_multiple_points() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let s0 = indices[0];
    let s1 = indices[3];
    let index_ptrs = [s0 as *const t4a_index, s1 as *const t4a_index];
    let values = [
        0usize, 0usize, // point 0
        1usize, 0usize, // point 1
        1usize, 1usize, // point 2
    ];
    let mut out_re = [0.0; 3];
    assert_eq!(
        t4a_treetn_evaluate(
            tt,
            index_ptrs.as_ptr(),
            2,
            values.as_ptr(),
            3,
            out_re.as_mut_ptr(),
            std::ptr::null_mut()
        ),
        T4A_SUCCESS
    );
    assert_eq!(out_re, [1.0, 2.0, 4.0]);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_evaluate_complex_requires_out_im_buffer() {
    let s = new_index(2);
    let index_ptrs = [s as *const t4a_index];
    let interleaved = [1.0, 2.0, 0.0, 0.0];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_c64(1, index_ptrs.as_ptr(), interleaved.as_ptr(), 2, &mut tensor),
        T4A_SUCCESS
    );

    let tt = new_treetn(&[tensor as *const t4a_tensor]);
    let values = [0usize];
    let mut out_re = 0.0;
    assert_eq!(
        t4a_treetn_evaluate(
            tt,
            index_ptrs.as_ptr(),
            1,
            values.as_ptr(),
            1,
            &mut out_re,
            std::ptr::null_mut()
        ),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_im is required"));

    t4a_treetn_release(tt);
    t4a_tensor_release(tensor);
    t4a_index_release(s);
}

#[test]
fn test_treetn_evaluate_complex_requires_out_im_even_for_real_value() {
    let s = new_index(2);
    let index_ptrs = [s as *const t4a_index];
    let interleaved = [1.0, 2.0, 3.0, 0.0];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_c64(1, index_ptrs.as_ptr(), interleaved.as_ptr(), 2, &mut tensor),
        T4A_SUCCESS
    );

    let tt = new_treetn(&[tensor as *const t4a_tensor]);
    let values = [1usize];
    let mut out_re = 0.0;
    assert_eq!(
        t4a_treetn_evaluate(
            tt,
            index_ptrs.as_ptr(),
            1,
            values.as_ptr(),
            1,
            &mut out_re,
            std::ptr::null_mut()
        ),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_im is required"));

    t4a_treetn_release(tt);
    t4a_tensor_release(tensor);
    t4a_index_release(s);
}

#[test]
fn test_treetn_evaluator_evaluate_requires_mutable_handle_signature() {
    let _evaluate: extern "C" fn(
        *mut t4a_treetn_evaluator,
        *const libc::size_t,
        libc::size_t,
        *mut libc::c_double,
        *mut libc::c_double,
    ) -> t4a_status_code = t4a_treetn_evaluator_evaluate;
}

#[test]
fn test_treetn_evaluator_reuses_index_order_for_multiple_batches() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let s0 = indices[0];
    let s1 = indices[3];
    let index_ptrs = [s0 as *const t4a_index, s1 as *const t4a_index];
    let mut evaluator = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_evaluator_new(tt, index_ptrs.as_ptr(), 2, &mut evaluator),
        T4A_SUCCESS
    );
    assert!(!evaluator.is_null());

    let first_values = [
        0usize, 0usize, // point 0
        1usize, 1usize, // point 1
    ];
    let mut first_out = [0.0; 2];
    assert_eq!(
        t4a_treetn_evaluator_evaluate(
            evaluator,
            first_values.as_ptr(),
            2,
            first_out.as_mut_ptr(),
            std::ptr::null_mut()
        ),
        T4A_SUCCESS
    );
    assert_eq!(first_out, [1.0, 4.0]);

    let second_values = [
        1usize, 0usize, // point 0
        0usize, 1usize, // point 1
    ];
    let mut second_out = [0.0; 2];
    assert_eq!(
        t4a_treetn_evaluator_evaluate(
            evaluator,
            second_values.as_ptr(),
            2,
            second_out.as_mut_ptr(),
            std::ptr::null_mut()
        ),
        T4A_SUCCESS
    );
    assert_eq!(second_out, [2.0, 3.0]);

    t4a_treetn_evaluator_release(evaluator);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_evaluator_owns_tree_after_original_release() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let s0 = indices[0];
    let s1 = indices[3];
    let index_ptrs = [s0 as *const t4a_index, s1 as *const t4a_index];
    let mut evaluator = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_evaluator_new(tt, index_ptrs.as_ptr(), 2, &mut evaluator),
        T4A_SUCCESS
    );
    assert!(!evaluator.is_null());

    t4a_treetn_release(tt);

    let values = [
        0usize, 0usize, // point 0
        1usize, 1usize, // point 1
    ];
    let mut out = [0.0; 2];
    assert_eq!(
        t4a_treetn_evaluator_evaluate(
            evaluator,
            values.as_ptr(),
            2,
            out.as_mut_ptr(),
            std::ptr::null_mut()
        ),
        T4A_SUCCESS
    );
    assert_eq!(out, [1.0, 4.0]);

    t4a_treetn_evaluator_release(evaluator);
    for tensor in tensors {
        t4a_tensor_release(tensor);
    }
    for index in indices {
        t4a_index_release(index);
    }
}

#[test]
fn test_treetn_evaluator_complex_requires_out_im_even_for_real_value() {
    let s = new_index(2);
    let index_ptrs = [s as *const t4a_index];
    let interleaved = [1.0, 2.0, 3.0, 0.0];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_c64(1, index_ptrs.as_ptr(), interleaved.as_ptr(), 2, &mut tensor),
        T4A_SUCCESS
    );
    let tt = new_treetn(&[tensor as *const t4a_tensor]);

    let mut evaluator = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_evaluator_new(tt, index_ptrs.as_ptr(), 1, &mut evaluator),
        T4A_SUCCESS
    );

    let values = [1usize];
    let mut out_re = 0.0;
    assert_eq!(
        t4a_treetn_evaluator_evaluate(
            evaluator,
            values.as_ptr(),
            1,
            &mut out_re,
            std::ptr::null_mut()
        ),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_im is required"));

    t4a_treetn_evaluator_release(evaluator);
    t4a_treetn_release(tt);
    t4a_tensor_release(tensor);
    t4a_index_release(s);
}

#[test]
fn test_treetn_evaluator_clone_remains_usable_after_original_release() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let s0 = indices[0];
    let s1 = indices[3];
    let index_ptrs = [s0 as *const t4a_index, s1 as *const t4a_index];
    let mut evaluator = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_evaluator_new(tt, index_ptrs.as_ptr(), 2, &mut evaluator),
        T4A_SUCCESS
    );
    assert_eq!(t4a_treetn_evaluator_is_assigned(evaluator), 1);
    assert_eq!(t4a_treetn_evaluator_is_assigned(std::ptr::null()), 0);

    let mut clone = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_evaluator_clone(evaluator, &mut clone),
        T4A_SUCCESS
    );
    assert!(!clone.is_null());
    t4a_treetn_evaluator_release(evaluator);

    let values = [
        0usize, 0usize, // point 0
        1usize, 1usize, // point 1
    ];
    let mut out = [0.0; 2];
    assert_eq!(
        t4a_treetn_evaluator_evaluate(
            clone,
            values.as_ptr(),
            2,
            out.as_mut_ptr(),
            std::ptr::null_mut()
        ),
        T4A_SUCCESS
    );
    assert_eq!(out, [1.0, 4.0]);

    t4a_treetn_evaluator_release(clone);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_evaluator_new_rejects_incomplete_index_list() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let s0 = indices[0];
    let index_ptrs = [s0 as *const t4a_index];
    let mut evaluator = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_evaluator_new(tt, index_ptrs.as_ptr(), 1, &mut evaluator),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("indices.len()"));
    assert!(evaluator.is_null());

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_contract_and_to_dense() {
    let in_idx = new_index(2);
    let out_idx = new_index(2);
    let mut in_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(in_idx, &mut in_clone),
        T4A_SUCCESS
    );

    let op_tensor = new_tensor(
        &[out_idx as *const t4a_index, in_idx as *const t4a_index],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let state_tensor = new_tensor(&[in_clone as *const t4a_index], &[5.0, 7.0]);
    let op = new_treetn(&[op_tensor as *const t4a_tensor]);
    let state = new_treetn(&[state_tensor as *const t4a_tensor]);

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_contract(
            op,
            state,
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            t4a_factorize_alg::SVD,
            0.0,
            4,
            &mut result
        ),
        T4A_SUCCESS
    );
    assert_eq!(read_dense_f64_treetn(result), vec![26.0, 38.0]);

    t4a_treetn_release(result);
    t4a_treetn_release(state);
    t4a_treetn_release(op);
    t4a_tensor_release(state_tensor);
    t4a_tensor_release(op_tensor);
    t4a_index_release(in_clone);
    t4a_index_release(out_idx);
    t4a_index_release(in_idx);
}

#[test]
fn test_treetn_contract_naive_requires_dense_limit() {
    let ((a, a_tensors, a_indices), (b, b_tensors, b_indices)) = make_two_site_contract_operands();

    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_contract(
            a,
            b,
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            t4a_factorize_alg::SVD,
            0.0,
            0,
            &mut out
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(out.is_null());
    assert!(last_error().contains("explicit dense/reference limit"));

    assert_eq!(
        t4a_treetn_contract(
            a,
            b,
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            t4a_factorize_alg::SVD,
            0.0,
            3,
            &mut out
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(out.is_null());
    assert!(last_error().contains("exceeding limit 3"));

    cleanup(a, a_tensors, a_indices);
    cleanup(b, b_tensors, b_indices);
}

#[test]
fn test_treetn_partial_contract_elementwise_diagonal_pairs() {
    let (a, a_tensors, a_indices) = make_two_site_treetn();
    let (b, b_tensors, b_indices) = make_two_site_treetn();

    let a_sites_0 = read_siteinds(a, 0);
    let a_sites_1 = read_siteinds(a, 1);
    let b_sites_0 = read_siteinds(b, 0);
    let b_sites_1 = read_siteinds(b, 1);

    let diagonal_left = [
        a_sites_0[0] as *const t4a_index,
        a_sites_1[0] as *const t4a_index,
    ];
    let diagonal_right = [
        b_sites_0[0] as *const t4a_index,
        b_sites_1[0] as *const t4a_index,
    ];
    let output_order = diagonal_left;

    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_partial_contract(
            a,
            b,
            0,
            std::ptr::null(),
            std::ptr::null(),
            diagonal_left.len(),
            diagonal_left.as_ptr(),
            diagonal_right.as_ptr(),
            output_order.len(),
            output_order.as_ptr(),
            0,
            t4a_contract_method::Zipup,
            std::ptr::null(),
            0,
            0,
            0.0,
            t4a_factorize_alg::SVD,
            0.0,
            0,
            &mut out,
        ),
        T4A_SUCCESS
    );

    assert_vec_close(&read_dense_f64_treetn(out), &[1.0, 4.0, 9.0, 16.0], 1e-12);

    t4a_treetn_release(out);
    for index in a_sites_0
        .into_iter()
        .chain(a_sites_1)
        .chain(b_sites_0)
        .chain(b_sites_1)
    {
        t4a_index_release(index);
    }
    cleanup(a, a_tensors, a_indices);
    cleanup(b, b_tensors, b_indices);
}

#[test]
fn test_treetn_contract_fit_zero_sweeps_uses_backend_default() {
    let ((a, a_tensors, a_indices), (b, b_tensors, b_indices)) = make_two_site_contract_operands();

    let mut baseline = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_contract(
            a,
            b,
            t4a_contract_method::Fit,
            std::ptr::null(),
            0,
            1,
            1e-30,
            t4a_factorize_alg::LU,
            0.0,
            0,
            &mut baseline
        ),
        T4A_SUCCESS
    );

    let mut uses_default = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_contract(
            a,
            b,
            t4a_contract_method::Fit,
            std::ptr::null(),
            0,
            0,
            1e-30,
            t4a_factorize_alg::LU,
            0.0,
            0,
            &mut uses_default
        ),
        T4A_SUCCESS
    );

    let baseline_dense = read_dense_f64_treetn(baseline);
    let uses_default_dense = read_dense_f64_treetn(uses_default);
    assert_vec_close(&uses_default_dense, &baseline_dense, 1e-10);

    t4a_treetn_release(uses_default);
    t4a_treetn_release(baseline);
    cleanup(a, a_tensors, a_indices);
    cleanup(b, b_tensors, b_indices);
}

#[test]
fn test_treetn_contract_fit_accepts_iterative_options() {
    let ((a, a_tensors, a_indices), (b, b_tensors, b_indices)) = make_two_site_contract_operands();

    let mut expected = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_contract(
            a,
            b,
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            t4a_factorize_alg::SVD,
            0.0,
            1024,
            &mut expected
        ),
        T4A_SUCCESS
    );

    let mut fitted = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_contract(
            a,
            b,
            t4a_contract_method::Fit,
            std::ptr::null(),
            0,
            2,
            1e-30,
            t4a_factorize_alg::LU,
            0.0,
            0,
            &mut fitted
        ),
        T4A_SUCCESS
    );

    let expected_dense = read_dense_f64_treetn(expected);
    let fitted_dense = read_dense_f64_treetn(fitted);
    assert_vec_close(&fitted_dense, &expected_dense, 1e-10);

    t4a_treetn_release(fitted);
    t4a_treetn_release(expected);
    cleanup(a, a_tensors, a_indices);
    cleanup(b, b_tensors, b_indices);
}

#[test]
fn test_treetn_apply_operator_chain_identity_single_site() {
    let state_site = new_index(2);
    let state = new_tensor(&[state_site as *const t4a_index], &[1.5, -2.0]);
    let state_tt = new_treetn(&[state as *const t4a_tensor]);

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [0usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            &mut result
        ),
        T4A_SUCCESS
    );
    assert_eq!(read_dense_f64_treetn(result), vec![1.5, -2.0]);

    let siteinds = read_siteinds(result, 0);
    assert_eq!(siteinds.len(), 1);
    let mut dim = 0usize;
    assert_eq!(
        crate::index::t4a_index_dim(siteinds[0], &mut dim),
        T4A_SUCCESS
    );
    assert_eq!(dim, 2);
    t4a_index_release(siteinds[0]);

    t4a_treetn_release(result);
    t4a_treetn_release(op_tt);
    t4a_treetn_release(state_tt);
    t4a_tensor_release(op);
    t4a_tensor_release(state);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    t4a_index_release(state_site);
}

#[test]
fn test_treetn_apply_operator_chain_partial_operator() {
    let (state_tt, state_tensors, mut state_indices) = make_two_site_treetn();

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[2.0, 0.0, 0.0, 5.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [1usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            &mut result
        ),
        T4A_SUCCESS
    );
    let dense = read_dense_f64_treetn(result);
    assert_vec_close(&dense, &[2.0, 4.0, 15.0, 20.0], 1e-10);

    t4a_treetn_release(result);
    t4a_treetn_release(op_tt);
    t4a_tensor_release(op);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    cleanup(state_tt, state_tensors, std::mem::take(&mut state_indices));
}

#[test]
fn test_treetn_apply_operator_chain_fit_zero_sweeps_uses_backend_default() {
    let (state_tt, state_tensors, mut state_indices) = make_two_site_treetn();

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[2.0, 0.0, 0.0, 5.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [1usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];
    let policy = relative_policy(1e-12);

    let mut baseline = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Fit,
            &policy,
            0,
            1,
            1e-30,
            &mut baseline
        ),
        T4A_SUCCESS
    );

    let mut uses_default = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Fit,
            &policy,
            0,
            0,
            1e-30,
            &mut uses_default
        ),
        T4A_SUCCESS
    );

    let baseline_dense = read_dense_f64_treetn(baseline);
    let uses_default_dense = read_dense_f64_treetn(uses_default);
    assert_vec_close(&uses_default_dense, &baseline_dense, 1e-10);

    t4a_treetn_release(uses_default);
    t4a_treetn_release(baseline);
    t4a_treetn_release(op_tt);
    t4a_tensor_release(op);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    cleanup(state_tt, state_tensors, std::mem::take(&mut state_indices));
}

#[test]
fn test_treetn_apply_operator_chain_fit_accepts_iterative_options() {
    let (state_tt, state_tensors, mut state_indices) = make_two_site_treetn();

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[2.0, 0.0, 0.0, 5.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [1usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];
    let policy = relative_policy(1e-12);

    let mut expected = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            &mut expected
        ),
        T4A_SUCCESS
    );

    let mut fitted = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Fit,
            &policy,
            0,
            2,
            1e-30,
            &mut fitted
        ),
        T4A_SUCCESS
    );

    let expected_dense = read_dense_f64_treetn(expected);
    let fitted_dense = read_dense_f64_treetn(fitted);
    assert_vec_close(&fitted_dense, &expected_dense, 1e-10);

    t4a_treetn_release(fitted);
    t4a_treetn_release(expected);
    t4a_treetn_release(op_tt);
    t4a_tensor_release(op);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    cleanup(state_tt, state_tensors, std::mem::take(&mut state_indices));
}

#[test]
fn test_treetn_apply_operator_chain_rejects_out_of_range_mapped_node() {
    let (state_tt, state_tensors, mut state_indices) = make_two_site_treetn();

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [2usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];
    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            &mut result
        ),
        T4A_INVALID_ARGUMENT
    );
    let err = last_error();
    assert!(
        err.contains("mapped node position 2"),
        "unexpected error: {err}"
    );
    assert!(result.is_null());

    t4a_treetn_release(op_tt);
    t4a_tensor_release(op);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    cleanup(state_tt, state_tensors, std::mem::take(&mut state_indices));
}

#[test]
fn test_treetn_apply_operator_chain_rejects_empty_mapping() {
    let state_site = new_index(2);
    let state = new_tensor(&[state_site as *const t4a_index], &[1.0, 2.0]);
    let state_tt = new_treetn(&[state as *const t4a_tensor]);

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes: [usize; 0] = [];
    let input_indices: [*const t4a_index; 0] = [];
    let output_indices: [*const t4a_index; 0] = [];
    let true_output_indices: [*const t4a_index; 0] = [];
    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            &mut result
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("mapped node positions must not be empty"));
    assert!(result.is_null());

    t4a_treetn_release(op_tt);
    t4a_treetn_release(state_tt);
    t4a_tensor_release(op);
    t4a_tensor_release(state);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    t4a_index_release(state_site);
}

#[test]
fn test_treetn_apply_operator_chain_rejects_unknown_internal_input_index() {
    let state_site = new_index(2);
    let state = new_tensor(&[state_site as *const t4a_index], &[1.0, 2.0]);
    let state_tt = new_treetn(&[state as *const t4a_tensor]);

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let wrong_in = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [0usize];
    let input_indices = [wrong_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];
    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            std::ptr::null(),
            0,
            1,
            0.0,
            &mut result
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("provided internal input index"));
    assert!(result.is_null());

    t4a_treetn_release(op_tt);
    t4a_treetn_release(state_tt);
    t4a_tensor_release(op);
    t4a_tensor_release(state);
    t4a_index_release(target_out);
    t4a_index_release(wrong_in);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    t4a_index_release(state_site);
}

#[test]
fn test_treetn_linsolve_two_site_identity_without_mapping_zero_sweeps() {
    let (rhs_tt, rhs_tensors, mut rhs_indices) = make_two_site_treetn();
    let expected = read_dense_f64_treetn(rhs_tt);

    let rhs_site0 = read_siteinds(rhs_tt, 0).remove(0);
    let rhs_site1 = read_siteinds(rhs_tt, 1).remove(0);
    let (op_tt, op_tensors, op_indices) =
        make_two_site_identity_operator_with_shared_inputs(rhs_site0, rhs_site1);

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_treetn_linsolve(
        op_tt,
        rhs_tt,
        rhs_tt,
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        0,
        0,
        1e-12,
        30,
        10,
        0.0,
        1.0,
        0.0,
        &mut result,
    );
    assert_eq!(status, T4A_SUCCESS, "{}", last_error());
    assert_vec_close(&read_dense_f64_treetn(result), &expected, 1e-10);

    t4a_treetn_release(result);
    t4a_treetn_release(op_tt);
    for tensor in op_tensors {
        t4a_tensor_release(tensor);
    }
    for index in op_indices {
        t4a_index_release(index);
    }
    t4a_index_release(rhs_site1);
    t4a_index_release(rhs_site0);
    cleanup(rhs_tt, rhs_tensors, std::mem::take(&mut rhs_indices));
}

#[test]
fn test_treetn_linsolve_two_site_identity_with_explicit_mapping() {
    let (rhs_tt, rhs_tensors, mut rhs_indices) = make_two_site_treetn();
    let expected = read_dense_f64_treetn(rhs_tt);

    let rhs_site0 = read_siteinds(rhs_tt, 0).remove(0);
    let rhs_site1 = read_siteinds(rhs_tt, 1).remove(0);
    let (op_tt, op_tensors, op_indices, internal_inputs, internal_outputs) =
        make_two_site_identity_operator_with_internal_indices();

    let mapped_vertices = [0usize, 1usize];
    let true_inputs = [rhs_site0 as *const t4a_index, rhs_site1 as *const t4a_index];
    let internal_inputs = [
        internal_inputs[0] as *const t4a_index,
        internal_inputs[1] as *const t4a_index,
    ];
    let true_outputs = [rhs_site0 as *const t4a_index, rhs_site1 as *const t4a_index];
    let internal_outputs = [
        internal_outputs[0] as *const t4a_index,
        internal_outputs[1] as *const t4a_index,
    ];

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    let status = t4a_treetn_linsolve(
        op_tt,
        rhs_tt,
        rhs_tt,
        0,
        mapped_vertices.as_ptr(),
        mapped_vertices.len(),
        true_inputs.as_ptr(),
        internal_inputs.as_ptr(),
        true_outputs.as_ptr(),
        internal_outputs.as_ptr(),
        std::ptr::null(),
        0,
        3,
        1e-12,
        30,
        10,
        0.0,
        1.0,
        0.0,
        &mut result,
    );
    assert_eq!(status, T4A_SUCCESS, "{}", last_error());
    assert_vec_close(&read_dense_f64_treetn(result), &expected, 1e-10);

    t4a_treetn_release(result);
    t4a_treetn_release(op_tt);
    for tensor in op_tensors {
        t4a_tensor_release(tensor);
    }
    for index in op_indices {
        t4a_index_release(index);
    }
    t4a_index_release(rhs_site1);
    t4a_index_release(rhs_site0);
    cleanup(rhs_tt, rhs_tensors, std::mem::take(&mut rhs_indices));
}

#[test]
fn test_treetn_linsolve_rejects_mapping_true_output_not_on_rhs() {
    let true_input = new_index(2);
    let true_output = new_index(2);
    let wrong_output = new_index(2);
    let internal_input = new_index(2);
    let internal_output = new_index(2);

    let init = new_tensor(&[true_input as *const t4a_index], &[1.0, 1.0]);
    let rhs = new_tensor(&[true_output as *const t4a_index], &[3.0, 4.0]);
    let init_tt = new_treetn(&[init as *const t4a_tensor]);
    let rhs_tt = new_treetn(&[rhs as *const t4a_tensor]);

    let op = new_tensor(
        &[
            internal_output as *const t4a_index,
            internal_input as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_vertices = [0usize];
    let true_inputs = [true_input as *const t4a_index];
    let internal_inputs = [internal_input as *const t4a_index];
    let true_outputs = [wrong_output as *const t4a_index];
    let internal_outputs = [internal_output as *const t4a_index];

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_linsolve(
            op_tt,
            rhs_tt,
            init_tt,
            0,
            mapped_vertices.as_ptr(),
            mapped_vertices.len(),
            true_inputs.as_ptr(),
            internal_inputs.as_ptr(),
            true_outputs.as_ptr(),
            internal_outputs.as_ptr(),
            std::ptr::null(),
            0,
            4,
            1e-12,
            30,
            10,
            0.0,
            1.0,
            0.0,
            &mut result
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("true output"));
    assert!(result.is_null());

    t4a_treetn_release(op_tt);
    t4a_treetn_release(rhs_tt);
    t4a_treetn_release(init_tt);
    t4a_tensor_release(op);
    t4a_tensor_release(rhs);
    t4a_tensor_release(init);
    t4a_index_release(internal_output);
    t4a_index_release(internal_input);
    t4a_index_release(wrong_output);
    t4a_index_release(true_output);
    t4a_index_release(true_input);
}
