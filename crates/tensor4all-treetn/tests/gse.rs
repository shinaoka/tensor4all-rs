use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IndexLike, TensorContractionLike, TensorDynLen};
use tensor4all_treetn::{
    global_subspace_expand, global_subspace_expand_with_references, gse_tdvp, GseError, GseOptions,
    GseTdvpOptions, IndexMapping, LinearOperator, TdvpOptions, TreeTN,
};

fn product_chain_state(
    amplitudes: [[Complex64; 2]; 2],
) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 2]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), bond.clone()],
        vec![amplitudes[0][0], amplitudes[0][1]],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![amplitudes[1][0], amplitudes[1][1]],
    )
    .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    state.connect(n0, &bond, n1, &bond).unwrap();
    (state, [s0, s1])
}

fn product_chain_state_f64(
    amplitudes: [[f64; 2]; 2],
) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 2]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), bond.clone()],
        vec![amplitudes[0][0], amplitudes[0][1]],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![amplitudes[1][0], amplitudes[1][1]],
    )
    .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    state.connect(n0, &bond, n1, &bond).unwrap();
    (state, [s0, s1])
}

fn product_chain_state_from_vectors(
    left_amplitudes: &[Complex64],
    right_amplitudes: &[Complex64],
) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 2]) {
    let s0 = DynIndex::new_dyn(left_amplitudes.len());
    let s1 = DynIndex::new_dyn(right_amplitudes.len());
    let bond = DynIndex::new_dyn(1);
    let t0 =
        TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], left_amplitudes.to_vec()).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond.clone(), s1.clone()], right_amplitudes.to_vec())
        .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    state.connect(n0, &bond, n1, &bond).unwrap();
    (state, [s0, s1])
}

fn product_chain3_state(
    amplitudes: [[Complex64; 2]; 3],
) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 3]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(1);
    let b12 = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), b01.clone()],
        vec![amplitudes[0][0], amplitudes[0][1]],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![b01.clone(), s1.clone(), b12.clone()],
        vec![amplitudes[1][0], amplitudes[1][1]],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(
        vec![b12.clone(), s2.clone()],
        vec![amplitudes[2][0], amplitudes[2][1]],
    )
    .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    let n2 = state.add_tensor("site2", t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();
    (state, [s0, s1, s2])
}

fn product_star_state(
    amplitudes: [[Complex64; 2]; 3],
) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 3]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(1);
    let b02 = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), b01.clone(), b02.clone()],
        vec![amplitudes[0][0], amplitudes[0][1]],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![b01.clone(), s1.clone()],
        vec![amplitudes[1][0], amplitudes[1][1]],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(
        vec![b02.clone(), s2.clone()],
        vec![amplitudes[2][0], amplitudes[2][1]],
    )
    .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    let n2 = state.add_tensor("site2", t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n0, &b02, n2, &b02).unwrap();
    (state, [s0, s1, s2])
}

fn product_internal_branch_state(
    amplitudes: [[Complex64; 2]; 4],
) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 4]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(1);
    let b12 = DynIndex::new_dyn(1);
    let b13 = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), b01.clone()],
        vec![amplitudes[0][0], amplitudes[0][1]],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![b01.clone(), s1.clone(), b12.clone(), b13.clone()],
        vec![amplitudes[1][0], amplitudes[1][1]],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(
        vec![b12.clone(), s2.clone()],
        vec![amplitudes[2][0], amplitudes[2][1]],
    )
    .unwrap();
    let t3 = TensorDynLen::from_dense(
        vec![b13.clone(), s3.clone()],
        vec![amplitudes[3][0], amplitudes[3][1]],
    )
    .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    let n2 = state.add_tensor("site2", t2).unwrap();
    let n3 = state.add_tensor("site3", t3).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();
    state.connect(n1, &b13, n3, &b13).unwrap();
    (state, [s0, s1, s2, s3])
}

fn entangled_chain3_state(
    leaf_matrix: [[Complex64; 2]; 2],
) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 3]) {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(1);
    let b12 = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![one, zero]).unwrap();

    let mut t1_data = vec![zero; 4];
    for b in 0..2 {
        t1_data[b + 2 * b] = one;
    }
    let t1 = TensorDynLen::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], t1_data).unwrap();

    let mut t2_data = vec![zero; 4];
    for b in 0..2 {
        for s in 0..2 {
            t2_data[b + 2 * s] = leaf_matrix[b][s];
        }
    }
    let t2 = TensorDynLen::from_dense(vec![b12.clone(), s2.clone()], t2_data).unwrap();

    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    let n2 = state.add_tensor("site2", t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();
    (state, [s0, s1, s2])
}

fn identity_or_x_operator(
    state_sites: &[DynIndex],
    node_names: &[&'static str],
    edges: &[(&'static str, &'static str)],
    x_nodes: &[&'static str],
) -> LinearOperator<TensorDynLen, &'static str> {
    let mut node_indices = HashMap::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let mut op_bonds = HashMap::new();

    for &(a, b) in edges {
        op_bonds.insert((a, b), DynIndex::new_dyn(1));
    }

    let mut mpo = TreeTN::<TensorDynLen, &'static str>::new();
    for (i, &name) in node_names.iter().enumerate() {
        let input = DynIndex::new_dyn(2);
        let output = DynIndex::new_dyn(2);
        input_mapping.insert(
            name,
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: input.clone(),
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: output.clone(),
            },
        );

        let mut inds = Vec::new();
        for &(a, b) in edges {
            if a == name || b == name {
                inds.push(op_bonds.get(&(a, b)).unwrap().clone());
            }
        }
        inds.push(output.clone());
        inds.push(input.clone());
        let dims: Vec<_> = inds.iter().map(DynIndex::dim).collect();
        let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
        let output_pos = inds.len() - 2;
        let input_pos = inds.len() - 1;
        for input_bit in 0..2 {
            let output_bit = if x_nodes.contains(&name) {
                1 - input_bit
            } else {
                input_bit
            };
            let mut coord = vec![0; inds.len()];
            coord[output_pos] = output_bit;
            coord[input_pos] = input_bit;
            data[col_major_offset(&coord, &dims)] = Complex64::new(1.0, 0.0);
        }
        let node = mpo
            .add_tensor(name, TensorDynLen::from_dense(inds, data).unwrap())
            .unwrap();
        node_indices.insert(name, node);
    }

    for &(a, b) in edges {
        let bond = op_bonds.get(&(a, b)).unwrap();
        mpo.connect(
            *node_indices.get(a).unwrap(),
            bond,
            *node_indices.get(b).unwrap(),
            bond,
        )
        .unwrap();
    }

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn col_major_offset(coord: &[usize], dims: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&value, &dim) in coord.iter().zip(dims) {
        offset += value * stride;
        stride *= dim;
    }
    offset
}

fn dense_distance(
    lhs: &TreeTN<TensorDynLen, &'static str>,
    rhs: &TreeTN<TensorDynLen, &'static str>,
) -> f64 {
    lhs.contract_to_tensor()
        .unwrap()
        .distance(&rhs.contract_to_tensor().unwrap())
        .unwrap()
}

fn edge_dim(state: &TreeTN<TensorDynLen, &'static str>, a: &str, b: &str) -> usize {
    let edge = state.edge_between(&a, &b).unwrap();
    state.bond_index(edge).unwrap().dim()
}

fn enable_grad_all(
    mut state: TreeTN<TensorDynLen, &'static str>,
) -> TreeTN<TensorDynLen, &'static str> {
    let nodes = state.node_indices().to_vec();
    for node in nodes {
        let tensor = state.tensor(node).unwrap().clone().enable_grad().unwrap();
        state.replace_tensor(node, tensor).unwrap();
    }
    state
}

fn local_basis_matrix(
    state: &TreeTN<TensorDynLen, &'static str>,
    parent: &str,
    child: &str,
    q_indices: &[DynIndex],
) -> (usize, usize, Vec<Complex64>) {
    let edge = state.edge_between(&parent, &child).unwrap();
    let bond = state.bond_index(edge).unwrap().clone();
    let child_tensor = state.tensor(state.node_index(&child).unwrap()).unwrap();
    let ordered = std::iter::once(bond.clone())
        .chain(q_indices.iter().cloned())
        .collect::<Vec<_>>();
    let data = child_tensor
        .permuteinds(&ordered)
        .unwrap()
        .to_vec::<Complex64>()
        .unwrap();
    let q_dim = q_indices.iter().map(DynIndex::dim).product();
    (bond.dim(), q_dim, data)
}

fn assert_rows_are_isometric(row_dim: usize, q_dim: usize, data: &[Complex64]) {
    for lhs in 0..row_dim {
        for rhs in 0..row_dim {
            let mut overlap = Complex64::new(0.0, 0.0);
            for q in 0..q_dim {
                overlap += data[lhs + row_dim * q] * data[rhs + row_dim * q].conj();
            }
            let expected = if lhs == rhs { 1.0 } else { 0.0 };
            assert!(
                (overlap - Complex64::new(expected, 0.0)).norm() < 1.0e-10,
                "basis rows {lhs},{rhs} overlap {overlap:?}, expected {expected}"
            );
        }
    }
}

fn projected_weight(row_dim: usize, q_dim: usize, data: &[Complex64], vector: &[Complex64]) -> f64 {
    let norm: f64 = vector.iter().map(Complex64::norm_sqr).sum();
    let mut projected = 0.0;
    for row in 0..row_dim {
        let mut coeff = Complex64::new(0.0, 0.0);
        for q in 0..q_dim {
            coeff += data[row + row_dim * q].conj() * vector[q];
        }
        projected += coeff.norm_sqr();
    }
    projected / norm
}

#[test]
fn global_subspace_expand_preserves_state_and_grows_chain_bond() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, sites) = product_chain_state([[one, zero], [one, zero]]);
    let (reference, _) = product_chain_state([[one, zero], [zero, one]]);
    let initial_bond_dim = edge_dim(&state, "site0", "site1");

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(result.references_built, 1);
    assert_eq!(result.edges_processed, 1);
    assert!(result.bonds_expanded >= 1);
    assert_eq!(
        edge_dim(&result.state, "site0", "site1"),
        initial_bond_dim + 1
    );
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
    assert!(result.state.canonical_region().contains(&"site0"));

    let edge = result.state.edge_between(&"site0", &"site1").unwrap();
    let bond = result.state.bond_index(edge).unwrap().clone();
    let child = result
        .state
        .tensor(result.state.node_index(&"site1").unwrap())
        .unwrap();
    let child_basis = child.permuteinds(&[bond, sites[1].clone()]).unwrap();
    let child_basis_data = child_basis.to_vec::<Complex64>().unwrap();
    let missing_site_direction_norm =
        child_basis_data[2].norm_sqr() + child_basis_data[3].norm_sqr();
    assert!(missing_site_direction_norm > 1.0 - 1.0e-10);
}

#[test]
fn global_subspace_expand_supports_real_tensors() {
    let (state, _) = product_chain_state_f64([[1.0, 0.0], [1.0, 0.0]]);
    let (reference, _) = product_chain_state_f64([[1.0, 0.0], [0.0, 1.0]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
}

#[test]
fn global_subspace_expand_maps_processed_child_bond_in_chain_q_space() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) = product_chain3_state([[one, zero], [one, zero], [one, zero]]);
    let (ref_tail, _) = product_chain3_state([[one, zero], [one, zero], [zero, one]]);
    let (ref_middle_tail, _) = product_chain3_state([[one, zero], [zero, one], [zero, one]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![ref_tail, ref_middle_tail],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(result.edges_processed, 2);
    assert!(result.bonds_expanded >= 2);
    assert_eq!(edge_dim(&result.state, "site1", "site2"), 2);
    assert!(edge_dim(&result.state, "site0", "site1") >= 2);
    assert!(result.state.same_topology(&state));
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
    assert!(result.state.canonical_region().contains(&"site0"));
}

#[test]
fn global_subspace_expand_handles_nonproduct_processed_child_bond_basis() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let half = Complex64::new(0.5, 0.0);
    let (state, sites) = entangled_chain3_state([[one, zero], [zero, half]]);
    let (reference, _) = entangled_chain3_state([[zero, one], [half, zero]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(result.edges_processed, 2);
    assert!(result.bonds_expanded >= 1);
    assert_eq!(edge_dim(&result.state, "site1", "site2"), 2);
    assert!(edge_dim(&result.state, "site0", "site1") >= 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);

    let child_edge = result.state.edge_between(&"site1", &"site2").unwrap();
    let processed_child_bond = result.state.bond_index(child_edge).unwrap().clone();
    let (row_dim, q_dim, basis) = local_basis_matrix(
        &result.state,
        "site0",
        "site1",
        &[sites[1].clone(), processed_child_bond],
    );
    assert_rows_are_isometric(row_dim, q_dim, &basis);

    let reference_q_direction = vec![zero, half, one, zero];
    assert!(projected_weight(row_dim, q_dim, &basis, &reference_q_direction) > 1.0 - 1.0e-10);
}

#[test]
fn global_subspace_expand_handles_branching_tree() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) = product_star_state([[one, zero], [one, zero], [one, zero]]);
    let (ref_leaf1, _) = product_star_state([[one, zero], [zero, one], [one, zero]]);
    let (ref_leaf2, _) = product_star_state([[one, zero], [one, zero], [zero, one]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![ref_leaf1, ref_leaf2],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(result.edges_processed, 2);
    assert!(result.bonds_expanded >= 2);
    assert!(result.state.same_topology(&state));
    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert_eq!(edge_dim(&result.state, "site0", "site2"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
    assert!(result.state.canonical_region().contains(&"site0"));
}

#[test]
fn global_subspace_expand_maps_two_processed_child_bonds_at_internal_node() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) =
        product_internal_branch_state([[one, zero], [one, zero], [one, zero], [one, zero]]);
    let (ref_leaf2, _) =
        product_internal_branch_state([[one, zero], [one, zero], [zero, one], [one, zero]]);
    let (ref_leaf3, _) =
        product_internal_branch_state([[one, zero], [one, zero], [one, zero], [zero, one]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![ref_leaf2, ref_leaf3],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(result.edges_processed, 3);
    assert!(result.bonds_expanded >= 3);
    assert_eq!(edge_dim(&result.state, "site1", "site2"), 2);
    assert_eq!(edge_dim(&result.state, "site1", "site3"), 2);
    assert!(edge_dim(&result.state, "site0", "site1") >= 2);
    assert!(result.state.same_topology(&state));
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
    assert!(result.state.canonical_region().contains(&"site0"));
}

#[test]
fn global_subspace_expand_builds_operator_references() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, sites) = product_chain_state([[one, zero], [one, zero]]);
    let operator = identity_or_x_operator(
        &sites,
        &["site0", "site1"],
        &[("site0", "site1")],
        &["site1"],
    );

    let result = global_subspace_expand(
        &operator,
        state.clone(),
        &"site0",
        GseOptions::default()
            .with_krylov_dim(1)
            .with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(result.references_built, 1);
    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
}

#[test]
fn global_subspace_expand_preserves_complex_phase_state() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let inv_sqrt2 = 0.5_f64.sqrt();
    let plus_i = Complex64::new(0.0, inv_sqrt2);
    let minus_i = Complex64::new(0.0, -inv_sqrt2);
    let (state, _) = product_chain_state([[Complex64::new(inv_sqrt2, 0.0), plus_i], [one, zero]]);
    let (reference, _) =
        product_chain_state([[Complex64::new(inv_sqrt2, 0.0), minus_i], [one, zero]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site1",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
    assert!(result.state.canonical_region().contains(&"site1"));
}

#[test]
fn global_subspace_expand_preserves_complex_reference_direction_in_larger_q_space() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let inv_sqrt2 = 0.5_f64.sqrt();
    let target_left = vec![one, zero, zero, zero];
    let reference_left = vec![
        zero,
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(0.0, inv_sqrt2),
        zero,
    ];
    let right = vec![one, zero];
    let (state, sites) = product_chain_state_from_vectors(&target_left, &right);
    let (reference, _) = product_chain_state_from_vectors(&reference_left, &right);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site1",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);

    let (row_dim, q_dim, basis) =
        local_basis_matrix(&result.state, "site1", "site0", &[sites[0].clone()]);
    assert_eq!(row_dim, 2);
    assert_eq!(q_dim, 4);
    assert_rows_are_isometric(row_dim, q_dim, &basis);
    assert!(
        projected_weight(row_dim, q_dim, &basis, &reference_left) > 1.0 - 1.0e-10,
        "expanded basis failed to preserve the complex reference direction"
    );
}

#[test]
fn global_subspace_expand_preserves_ad_tracking_through_local_density_path() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) = product_chain_state([[one, zero], [one, zero]]);
    let (reference, _) = product_chain_state([[one, zero], [zero, one]]);

    let result = global_subspace_expand_with_references(
        enable_grad_all(state),
        vec![enable_grad_all(reference)],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert!(result.state.node_indices().iter().all(|&node| result
        .state
        .tensor(node)
        .unwrap()
        .tracks_grad()));

    let loss = result.state.contract_to_tensor().unwrap().sum().unwrap();
    loss.backward().unwrap();
    for node in result.state.node_indices() {
        assert!(
            result.state.tensor(node).unwrap().grad().unwrap().is_some(),
            "expanded node {node:?} lost gradient tracking"
        );
    }
}

#[test]
fn global_subspace_expand_respects_density_weight_cutoff_without_dropping_target() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) = product_chain_state([[one, zero], [one, zero]]);
    let (reference, _) = product_chain_state([[one, zero], [zero, one]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(2.0),
    )
    .unwrap();

    assert_eq!(result.bonds_expanded, 0);
    assert_eq!(edge_dim(&result.state, "site0", "site1"), 1);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
}

#[test]
fn global_subspace_expand_with_empty_references_is_noop() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) = product_chain3_state([[one, zero], [one, zero], [one, zero]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        Vec::new(),
        &"site0",
        GseOptions::default(),
    )
    .unwrap();

    assert_eq!(result.references_built, 0);
    assert_eq!(result.edges_processed, 0);
    assert_eq!(result.bonds_expanded, 0);
    assert_eq!(result.max_added_basis, 0);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
    assert!(result.state.canonical_region().contains(&"site0"));
}

#[test]
fn global_subspace_expand_builds_multiple_krylov_references() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, sites) = product_chain_state([[one, zero], [one, zero]]);
    let operator = identity_or_x_operator(
        &sites,
        &["site0", "site1"],
        &[("site0", "site1")],
        &["site1"],
    );

    let result = global_subspace_expand(
        &operator,
        state.clone(),
        &"site0",
        GseOptions::default()
            .with_krylov_dim(2)
            .with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(result.references_built, 2);
    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
}

#[test]
fn gse_tdvp_runs_existing_tdvp_after_expansion() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, sites) = product_chain_state([[one, zero], [one, zero]]);
    let operator = identity_or_x_operator(
        &sites,
        &["site0", "site1"],
        &[("site0", "site1")],
        &["site1"],
    );

    let result = gse_tdvp(
        &operator,
        state,
        &"site0",
        GseTdvpOptions {
            gse: GseOptions::default()
                .with_krylov_dim(1)
                .with_density_weight_cutoff(1.0e-14),
            tdvp: TdvpOptions::default()
                .with_nsite(1)
                .with_nsweeps(1)
                .with_exponent_step(Complex64::new(0.0, -0.01)),
        },
    )
    .unwrap();

    assert_eq!(result.gse_expansions, 1);
    assert_eq!(result.sweeps_completed, 1);
    assert!(result.local_updates > 0);
    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
}

#[test]
fn gse_tdvp_can_skip_first_expansion_and_expand_later_sweeps() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, sites) = product_chain_state([[one, zero], [one, zero]]);
    let operator = identity_or_x_operator(
        &sites,
        &["site0", "site1"],
        &[("site0", "site1")],
        &["site1"],
    );

    let result = gse_tdvp(
        &operator,
        state,
        &"site0",
        GseTdvpOptions {
            gse: GseOptions::default()
                .with_krylov_dim(1)
                .with_density_weight_cutoff(1.0e-14)
                .with_expand_before_first_sweep(false),
            tdvp: TdvpOptions::default()
                .with_nsite(1)
                .with_nsweeps(2)
                .with_exponent_step(Complex64::new(0.0, -0.01)),
        },
    )
    .unwrap();

    assert_eq!(result.gse_expansions, 1);
    assert_eq!(result.sweeps_completed, 2);
    assert!(result.local_updates > 0);
    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
}

#[test]
fn gse_rejects_invalid_options_and_missing_center() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) = product_chain_state([[one, zero], [one, zero]]);

    let density_err = global_subspace_expand_with_references(
        state.clone(),
        Vec::new(),
        &"site0",
        GseOptions::default().with_density_weight_cutoff(-1.0),
    )
    .unwrap_err();
    assert!(matches!(
        density_err,
        GseError::InvalidOption {
            option: "density_weight_cutoff",
            ..
        }
    ));

    let hermitian_err = global_subspace_expand_with_references(
        state.clone(),
        Vec::new(),
        &"site0",
        GseOptions::default().with_hermitian_tol(-1.0),
    )
    .unwrap_err();
    assert!(matches!(
        hermitian_err,
        GseError::InvalidOption {
            option: "hermitian_tol",
            ..
        }
    ));

    let invalid_rank_options = GseOptions {
        reference_max_bond_dim: Some(0),
        ..Default::default()
    };
    let rank_err = global_subspace_expand_with_references(
        state.clone(),
        Vec::new(),
        &"site0",
        invalid_rank_options,
    )
    .unwrap_err();
    assert!(matches!(
        rank_err,
        GseError::InvalidOption {
            option: "reference_max_bond_dim",
            ..
        }
    ));

    let missing_err = global_subspace_expand_with_references(
        state,
        Vec::new(),
        &"missing",
        GseOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        missing_err,
        GseError::MissingCenter { center } if center == "\"missing\""
    ));
}

#[test]
fn gse_rejects_multiple_state_sites_per_node() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![s0, s1], vec![Complex64::new(0.0, 0.0); 4]).unwrap();
    let state =
        TreeTN::<TensorDynLen, &'static str>::from_tensors(vec![tensor], vec!["site0"]).unwrap();

    let err =
        global_subspace_expand_with_references(state, Vec::new(), &"site0", GseOptions::default())
            .unwrap_err();

    assert!(matches!(
        err,
        GseError::UnsupportedStateSiteCount { node, count }
            if node == "\"site0\"" && count == 2
    ));
}

#[test]
fn gse_rejects_reference_topology_mismatch() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (state, _) = product_chain_state([[one, zero], [one, zero]]);
    let (reference, _) = product_star_state([[one, zero], [one, zero], [one, zero]]);

    let err = global_subspace_expand_with_references(
        state,
        vec![reference],
        &"site0",
        GseOptions::default(),
    )
    .unwrap_err();

    assert!(matches!(err, GseError::TopologyMismatch));
}

#[test]
fn global_subspace_expand_accepts_complex_reference_for_real_target() {
    let (state, _) = product_chain_state_f64([[1.0, 0.0], [1.0, 0.0]]);
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (reference, _) = product_chain_state([[one, zero], [zero, one]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
}

#[test]
fn global_subspace_expand_accepts_mixed_scalar_storage_inside_target_tree() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 0.0]).unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    )
    .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    state.connect(n0, &bond, n1, &bond).unwrap();
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let (reference, _) = product_chain_state([[one, zero], [zero, one]]);

    let result = global_subspace_expand_with_references(
        state.clone(),
        vec![reference],
        &"site0",
        GseOptions::default().with_density_weight_cutoff(1.0e-14),
    )
    .unwrap();

    assert_eq!(edge_dim(&result.state, "site0", "site1"), 2);
    assert!(dense_distance(&result.state, &state) < 1.0e-10);
}
