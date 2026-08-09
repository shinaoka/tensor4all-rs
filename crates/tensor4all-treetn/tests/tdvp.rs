use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::krylov::HermitianKrylovExpmOptions;
use tensor4all_core::{
    DynIndex, FactorizeOptions, IndexLike, SvdTruncationPolicy, TensorContractionLike,
    TensorDynLen, TensorIndex,
};
use tensor4all_tensorbackend::{hermitian_eigendecomposition, Matrix};
use tensor4all_treetn::{
    factorize_tensor_to_treetn_with, tdvp, tdvp_with_treetn_operator, IndexMapping, LinearOperator,
    TdvpError, TdvpOptions, TreeTN, TreeTopology,
};

fn chain_state() -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 2]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), bond.clone()],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![Complex64::new(0.5, 0.0), Complex64::new(-1.0, 0.0)],
    )
    .unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    state.connect(n0, &bond, n1, &bond).unwrap();
    (state, [s0, s1])
}

fn star_state() -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 3]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(1);
    let b02 = DynIndex::new_dyn(1);
    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), b01.clone(), b02.clone()],
        vec![Complex64::new(1.0, 0.0), Complex64::new(-0.25, 0.0)],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![b01.clone(), s1.clone()],
        vec![Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(
        vec![b02.clone(), s2.clone()],
        vec![Complex64::new(-2.0, 0.0), Complex64::new(0.75, 0.0)],
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

fn identity_operator(
    state_sites: &[DynIndex],
    node_names: &[&'static str],
    edges: &[(&'static str, &'static str)],
) -> LinearOperator<TensorDynLen, &'static str> {
    let mut tensors = HashMap::new();
    let mut node_indices = HashMap::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let mut op_bonds = HashMap::new();

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
        tensors.insert(name, (input, output));
    }

    for &(a, b) in edges {
        op_bonds.insert((a, b), DynIndex::new_dyn(1));
    }

    let mut mpo = TreeTN::<TensorDynLen, &'static str>::new();
    for &name in node_names {
        let (input, output) = tensors.get(name).unwrap();
        let mut inds = Vec::new();
        for &(a, b) in edges {
            if a == name || b == name {
                inds.push(op_bonds.get(&(a, b)).unwrap().clone());
            }
        }
        inds.push(output.clone());
        inds.push(input.clone());
        let mut data = vec![Complex64::new(0.0, 0.0); inds.iter().map(DynIndex::dim).product()];
        let output_pos = inds.len() - 2;
        let input_pos = inds.len() - 1;
        let dims: Vec<_> = inds.iter().map(DynIndex::dim).collect();
        for s in 0..2 {
            let mut coord = vec![0; inds.len()];
            coord[output_pos] = s;
            coord[input_pos] = s;
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

fn diagonal_sum_z_operator(
    state_sites: &[DynIndex],
    node_names: &[&'static str],
    edges: &[(&'static str, &'static str)],
    coeffs: &[f64],
) -> LinearOperator<TensorDynLen, &'static str> {
    let mut dense_indices = Vec::new();
    let mut topology_nodes = HashMap::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for (i, &name) in node_names.iter().enumerate() {
        let output = DynIndex::new_dyn(2);
        let input = DynIndex::new_dyn(2);
        dense_indices.push(output.clone());
        dense_indices.push(input.clone());
        topology_nodes.insert(name, vec![output.clone(), input.clone()]);
        input_mapping.insert(
            name,
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: input,
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: output,
            },
        );
    }

    let dims = vec![2; dense_indices.len()];
    let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
    for basis in 0..(1usize << node_names.len()) {
        let mut coord = vec![0; dense_indices.len()];
        let mut energy = 0.0;
        for site in 0..node_names.len() {
            let bit = (basis >> site) & 1;
            coord[2 * site] = bit;
            coord[2 * site + 1] = bit;
            energy += coeffs[site] * if bit == 0 { 1.0 } else { -1.0 };
        }
        data[col_major_offset(&coord, &dims)] = Complex64::new(energy, 0.0);
    }

    let dense = TensorDynLen::from_dense(dense_indices, data).unwrap();
    let topology = TreeTopology::new(topology_nodes, edges.to_vec());
    let mpo = factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        node_names.first().unwrap(),
    )
    .unwrap();

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn heisenberg_operator(
    state_sites: &[DynIndex],
    node_names: &[&'static str],
    edges: &[(&'static str, &'static str)],
) -> LinearOperator<TensorDynLen, &'static str> {
    let mut dense_indices = Vec::new();
    let mut topology_nodes = HashMap::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for (i, &name) in node_names.iter().enumerate() {
        let output = DynIndex::new_dyn(2);
        let input = DynIndex::new_dyn(2);
        dense_indices.push(output.clone());
        dense_indices.push(input.clone());
        topology_nodes.insert(name, vec![output.clone(), input.clone()]);
        input_mapping.insert(
            name,
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: input,
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index: state_sites[i].clone(),
                internal_index: output,
            },
        );
    }

    let name_to_site: HashMap<_, _> = node_names
        .iter()
        .copied()
        .enumerate()
        .map(|(i, name)| (name, i))
        .collect();
    let dims = vec![2; dense_indices.len()];
    let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
    for input_basis in 0..(1usize << node_names.len()) {
        for &(left_name, right_name) in edges {
            let left = *name_to_site.get(left_name).unwrap();
            let right = *name_to_site.get(right_name).unwrap();
            let left_bit = (input_basis >> left) & 1;
            let right_bit = (input_basis >> right) & 1;
            let z_left = if left_bit == 0 { 1.0 } else { -1.0 };
            let z_right = if right_bit == 0 { 1.0 } else { -1.0 };
            add_dense_operator_entry(
                &mut data,
                &dims,
                node_names.len(),
                input_basis,
                input_basis,
                Complex64::new(z_left * z_right, 0.0),
            );

            if left_bit != right_bit {
                let flipped = input_basis ^ (1usize << left) ^ (1usize << right);
                add_dense_operator_entry(
                    &mut data,
                    &dims,
                    node_names.len(),
                    input_basis,
                    flipped,
                    Complex64::new(2.0, 0.0),
                );
            }
        }
    }

    let dense = TensorDynLen::from_dense(dense_indices, data).unwrap();
    let topology = TreeTopology::new(topology_nodes, edges.to_vec());
    let mpo = factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        node_names.first().unwrap(),
    )
    .unwrap();

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn add_dense_operator_entry(
    data: &mut [Complex64],
    dims: &[usize],
    n_sites: usize,
    input_basis: usize,
    output_basis: usize,
    value: Complex64,
) {
    let mut coord = vec![0; 2 * n_sites];
    for site in 0..n_sites {
        coord[2 * site] = (output_basis >> site) & 1;
        coord[2 * site + 1] = (input_basis >> site) & 1;
    }
    data[col_major_offset(&coord, dims)] += value;
}

fn dense_heisenberg_matrix(n_sites: usize, edges: &[(usize, usize)]) -> Matrix<Complex64> {
    let dim = 1usize << n_sites;
    let mut data = vec![Complex64::new(0.0, 0.0); dim * dim];
    for input_basis in 0..dim {
        for &(left, right) in edges {
            let left_bit = (input_basis >> left) & 1;
            let right_bit = (input_basis >> right) & 1;
            let z_left = if left_bit == 0 { 1.0 } else { -1.0 };
            let z_right = if right_bit == 0 { 1.0 } else { -1.0 };
            data[input_basis + dim * input_basis] += Complex64::new(z_left * z_right, 0.0);

            if left_bit != right_bit {
                let flipped = input_basis ^ (1usize << left) ^ (1usize << right);
                data[flipped + dim * input_basis] += Complex64::new(2.0, 0.0);
            }
        }
    }
    Matrix::from_col_major_vec(dim, dim, data)
}

fn exact_evolve(
    hamiltonian: &Matrix<Complex64>,
    initial: &[Complex64],
    time: f64,
) -> Vec<Complex64> {
    let decomp = hermitian_eigendecomposition(hamiltonian, 1.0e-12).unwrap();
    let n = hamiltonian.nrows();
    let vectors = decomp.eigenvectors.as_col_major_slice();
    let mut coefficients = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        for row in 0..n {
            coefficients[col] += vectors[row + col * n].conj() * initial[row];
        }
    }

    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        let phase = (Complex64::new(0.0, -time) * decomp.eigenvalues[col]).exp();
        let coeff = phase * coefficients[col];
        for row in 0..n {
            result[row] += vectors[row + col * n] * coeff;
        }
    }
    result
}

fn state_vector(state: &TreeTN<TensorDynLen, &'static str>, sites: &[DynIndex]) -> Vec<Complex64> {
    state
        .contract_to_tensor()
        .unwrap()
        .permuteinds(sites)
        .unwrap()
        .to_vec::<Complex64>()
        .unwrap()
}

fn l2_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (*a - *e).norm_sqr())
        .sum::<f64>()
        .sqrt()
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

fn assert_phase_evolution(
    before: &TensorDynLen,
    after: &TensorDynLen,
    exponent: Complex64,
    tolerance: f64,
) {
    let after = after.permuteinds(&before.external_indices()).unwrap();
    let before_data = before.to_vec::<Complex64>().unwrap();
    let after_data = after.to_vec::<Complex64>().unwrap();
    let phase = exponent.exp();
    let max_error = before_data
        .iter()
        .zip(after_data.iter())
        .map(|(x, y)| (*y - phase * *x).norm())
        .fold(0.0, f64::max);
    assert!(
        max_error < tolerance,
        "max phase evolution error {max_error:.3e} exceeds tolerance {tolerance:.3e}"
    );
}

fn assert_diagonal_sum_z_evolution(
    before: &TensorDynLen,
    after: &TensorDynLen,
    exponent: Complex64,
    coeffs: &[f64],
    tolerance: f64,
) {
    let after = after.permuteinds(&before.external_indices()).unwrap();
    let before_data = before.to_vec::<Complex64>().unwrap();
    let after_data = after.to_vec::<Complex64>().unwrap();
    let dims = vec![2; coeffs.len()];
    let max_error = before_data
        .iter()
        .zip(after_data.iter())
        .enumerate()
        .map(|(offset, (x, y))| {
            let coord = decode_col_major_offset(offset, &dims);
            let energy = coord
                .iter()
                .zip(coeffs.iter())
                .map(|(&bit, &coeff)| coeff * if bit == 0 { 1.0 } else { -1.0 })
                .sum::<f64>();
            (*y - (exponent * energy).exp() * *x).norm()
        })
        .fold(0.0, f64::max);
    assert!(
        max_error < tolerance,
        "max diagonal evolution error {max_error:.3e} exceeds tolerance {tolerance:.3e}"
    );
}

fn decode_col_major_offset(mut offset: usize, dims: &[usize]) -> Vec<usize> {
    let mut coord = Vec::with_capacity(dims.len());
    for &dim in dims {
        coord.push(offset % dim);
        offset /= dim;
    }
    coord
}

#[test]
fn two_site_tdvp_identity_chain_matches_global_phase() {
    let (state, sites) = chain_state();
    let before = state.contract_to_tensor().unwrap();
    let operator = identity_operator(&sites, &["site0", "site1"], &[("site0", "site1")]);
    let exponent = Complex64::new(0.0, -0.2);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(2)
            .with_order(2)
            .with_exponent_step(exponent),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 2);
    let after = result.state.contract_to_tensor().unwrap();
    assert_phase_evolution(&before, &after, exponent, 1.0e-10);
}

#[test]
fn two_site_tdvp_identity_star_matches_global_phase() {
    let (state, sites) = star_state();
    let before = state.contract_to_tensor().unwrap();
    let operator = identity_operator(
        &sites,
        &["site0", "site1", "site2"],
        &[("site0", "site1"), ("site0", "site2")],
    );
    let exponent = Complex64::new(0.0, -0.15);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(2)
            .with_order(2)
            .with_exponent_step(exponent),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 6);
    let after = result.state.contract_to_tensor().unwrap();
    assert_phase_evolution(&before, &after, exponent, 1.0e-10);
}

#[test]
fn two_site_tdvp_sum_z_chain_matches_exact_dense_phase() {
    let (state, sites) = chain_state();
    let before = state.contract_to_tensor().unwrap();
    let coeffs = [1.0, -0.35];
    let operator =
        diagonal_sum_z_operator(&sites, &["site0", "site1"], &[("site0", "site1")], &coeffs);
    let exponent = Complex64::new(0.0, -0.07);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(2)
            .with_order(4)
            .with_exponent_step(exponent),
    )
    .unwrap();

    let after = result.state.contract_to_tensor().unwrap();
    assert_diagonal_sum_z_evolution(&before, &after, exponent, &coeffs, 1.0e-10);
}

#[test]
fn two_site_tdvp_sum_z_star_matches_exact_dense_phase() {
    let (state, sites) = star_state();
    let before = state.contract_to_tensor().unwrap();
    let coeffs = [0.75, -0.5, 0.25];
    let operator = diagonal_sum_z_operator(
        &sites,
        &["site0", "site1", "site2"],
        &[("site0", "site1"), ("site0", "site2")],
        &coeffs,
    );
    let exponent = Complex64::new(0.0, -0.04);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(2)
            .with_order(2)
            .with_exponent_step(exponent),
    )
    .unwrap();

    let after = result.state.contract_to_tensor().unwrap();
    assert_diagonal_sum_z_evolution(&before, &after, exponent, &coeffs, 1.0e-10);
}

#[test]
fn one_site_tdvp_identity_single_node_matches_global_phase() {
    let site = DynIndex::new_dyn(2);
    let state_tensor = TensorDynLen::from_dense(
        vec![site.clone()],
        vec![Complex64::new(0.75, 0.0), Complex64::new(-1.25, 0.0)],
    )
    .unwrap();
    let state =
        TreeTN::<TensorDynLen, &'static str>::from_tensors(vec![state_tensor], vec!["site0"])
            .unwrap();
    let before = state.contract_to_tensor().unwrap();
    let operator = identity_operator(&[site], &["site0"], &[]);
    let exponent = Complex64::new(0.0, -0.3);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(1)
            .with_order(4)
            .with_exponent_step(exponent),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 6);
    let after = result.state.contract_to_tensor().unwrap();
    assert_phase_evolution(&before, &after, exponent, 1.0e-10);
}

#[test]
fn one_site_tdvp_identity_chain_matches_global_phase() {
    let (state, sites) = chain_state();
    let before = state.contract_to_tensor().unwrap();
    let operator = identity_operator(&sites, &["site0", "site1"], &[("site0", "site1")]);
    let exponent = Complex64::new(0.0, -0.12);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(1)
            .with_order(2)
            .with_exponent_step(exponent),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 6);
    let after = result.state.contract_to_tensor().unwrap();
    assert_phase_evolution(&before, &after, exponent, 1.0e-10);
}

#[test]
fn one_site_tdvp_identity_star_matches_global_phase() {
    let (state, sites) = star_state();
    let before = state.contract_to_tensor().unwrap();
    let operator = identity_operator(
        &sites,
        &["site0", "site1", "site2"],
        &[("site0", "site1"), ("site0", "site2")],
    );
    let exponent = Complex64::new(0.0, -0.08);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(1)
            .with_order(2)
            .with_exponent_step(exponent),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 10);
    let after = result.state.contract_to_tensor().unwrap();
    assert_phase_evolution(&before, &after, exponent, 1.0e-10);
}

#[test]
fn one_site_tdvp_sum_z_chain_preserves_norm_and_matches_exact_phase() {
    let (state, sites) = chain_state();
    let before = state.contract_to_tensor().unwrap();
    let before_norm = before.norm().unwrap();
    let coeffs = [0.4, -0.9];
    let operator =
        diagonal_sum_z_operator(&sites, &["site0", "site1"], &[("site0", "site1")], &coeffs);
    let exponent = Complex64::new(0.0, -0.05);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(1)
            .with_order(2)
            .with_exponent_step(exponent),
    )
    .unwrap();

    let after = result.state.contract_to_tensor().unwrap();
    assert!((after.norm().unwrap() - before_norm).abs() < 1.0e-10);
    assert_diagonal_sum_z_evolution(&before, &after, exponent, &coeffs, 1.0e-10);
}

#[test]
fn two_site_tdvp_heisenberg_chain_matches_dense_exact_evolution() {
    let (state, sites) = chain_state();
    let initial = state_vector(&state, &sites);
    let operator = heisenberg_operator(&sites, &["site0", "site1"], &[("site0", "site1")]);
    let dt = 0.03;
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(2)
            .with_order(2)
            .with_exponent_step(Complex64::new(0.0, -dt)),
    )
    .unwrap();

    let actual = state_vector(&result.state, &sites);
    let expected = exact_evolve(&dense_heisenberg_matrix(2, &[(0, 1)]), &initial, dt);
    let error = l2_error(&actual, &expected);
    assert!(
        error < 1.0e-9,
        "chain Heisenberg TDVP error {error:.3e} exceeds tolerance"
    );
}

#[test]
fn two_site_tdvp_heisenberg_star_tracks_dense_exact_evolution() {
    let (state, sites) = star_state();
    let initial = state_vector(&state, &sites);
    let operator = heisenberg_operator(
        &sites,
        &["site0", "site1", "site2"],
        &[("site0", "site1"), ("site0", "site2")],
    );
    let dt = 0.01;
    let result = tdvp(
        &operator,
        state,
        &"site1",
        TdvpOptions::default()
            .with_nsite(2)
            .with_order(2)
            .with_exponent_step(Complex64::new(0.0, -dt)),
    )
    .unwrap();

    let actual = state_vector(&result.state, &sites);
    let expected = exact_evolve(&dense_heisenberg_matrix(3, &[(0, 1), (0, 2)]), &initial, dt);
    let error = l2_error(&actual, &expected);
    assert!(
        error < 5.0e-6,
        "star Heisenberg TDVP error {error:.3e} exceeds tolerance"
    );
}

#[test]
fn one_site_tdvp_rejects_max_bond_dim_truncation() {
    let (state, sites) = chain_state();
    let operator = identity_operator(&sites, &["site0", "site1"], &[("site0", "site1")]);

    let err = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default().with_nsite(1).with_max_bond_dim(2),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        TdvpError::InvalidOption {
            option: "max_bond_dim",
            ..
        }
    ));
}

#[test]
fn one_site_tdvp_rejects_svd_truncation_policy() {
    let (state, sites) = chain_state();
    let operator = identity_operator(&sites, &["site0", "site1"], &[("site0", "site1")]);

    let err = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(1)
            .with_svd_policy(SvdTruncationPolicy::new(1.0e-12)),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        TdvpError::InvalidOption {
            option: "svd_policy",
            ..
        }
    ));
}

#[test]
fn tdvp_options_set_sweeps_and_krylov_options() {
    let krylov = HermitianKrylovExpmOptions {
        max_iter: 7,
        tol: 1.0e-11,
        ..Default::default()
    };
    let options = TdvpOptions::default()
        .with_nsweeps(2)
        .with_krylov_options(krylov);

    assert_eq!(options.nsweeps, 2);
    assert_eq!(options.krylov.max_iter, 7);
    assert_eq!(options.krylov.tol, 1.0e-11);
}

#[test]
fn tdvp_rejects_invalid_public_options() {
    let (state, sites) = chain_state();
    let operator = identity_operator(&sites, &["site0", "site1"], &[("site0", "site1")]);

    let err = tdvp(
        &operator,
        state.clone(),
        &"site0",
        TdvpOptions::default().with_nsite(3),
    )
    .unwrap_err();
    assert!(matches!(err, TdvpError::UnsupportedNsite { requested: 3 }));

    let err = tdvp(
        &operator,
        state.clone(),
        &"site0",
        TdvpOptions::default().with_nsweeps(0),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        TdvpError::InvalidOption {
            option: "nsweeps",
            ..
        }
    ));

    let err = tdvp(
        &operator,
        state.clone(),
        &"site0",
        TdvpOptions::default().with_order(3),
    )
    .unwrap_err();
    assert!(matches!(err, TdvpError::UnsupportedOrder { requested: 3 }));

    let err = tdvp(
        &operator,
        state.clone(),
        &"site0",
        TdvpOptions::default().with_exponent_step(Complex64::new(f64::NAN, 0.0)),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        TdvpError::InvalidOption {
            option: "exponent_step",
            ..
        }
    ));

    let err = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default().with_max_bond_dim(0),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        TdvpError::InvalidOption {
            option: "max_bond_dim",
            ..
        }
    ));
}

#[test]
fn tdvp_rejects_missing_center_and_topology_mismatch() {
    let (chain, chain_sites) = chain_state();
    let chain_operator =
        identity_operator(&chain_sites, &["site0", "site1"], &[("site0", "site1")]);
    let err = tdvp(
        &chain_operator,
        chain.clone(),
        &"missing",
        TdvpOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, TdvpError::MissingCenter { .. }));

    let (star, star_sites) = star_state();
    let star_operator = identity_operator(
        &star_sites,
        &["site0", "site1", "site2"],
        &[("site0", "site1"), ("site0", "site2")],
    );
    let err = tdvp(&star_operator, chain, &"site0", TdvpOptions::default()).unwrap_err();
    assert!(matches!(err, TdvpError::TopologyMismatch));

    let result = tdvp(&star_operator, star, &"site0", TdvpOptions::default());
    assert!(result.is_ok());
}

#[test]
fn two_site_tdvp_rejects_single_node_empty_sweep() {
    let site = DynIndex::new_dyn(2);
    let state_tensor = TensorDynLen::from_dense(
        vec![site.clone()],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    )
    .unwrap();
    let state =
        TreeTN::<TensorDynLen, &'static str>::from_tensors(vec![state_tensor], vec!["site0"])
            .unwrap();
    let operator = identity_operator(&[site], &["site0"], &[]);

    let err = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default().with_nsite(2),
    )
    .unwrap_err();

    assert!(matches!(err, TdvpError::EmptyTwoSiteSweep));
}

#[test]
fn two_site_tdvp_accepts_truncation_options() {
    let (state, sites) = chain_state();
    let operator = identity_operator(&sites, &["site0", "site1"], &[("site0", "site1")]);
    let result = tdvp(
        &operator,
        state,
        &"site0",
        TdvpOptions::default()
            .with_nsite(2)
            .with_max_bond_dim(1)
            .with_svd_policy(SvdTruncationPolicy::new(1.0e-12)),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 2);
}

#[test]
fn tdvp_with_treetn_operator_single_node_identity_runs() {
    let site = DynIndex::new_dyn(2);
    let state_tensor = TensorDynLen::from_dense(
        vec![site.clone()],
        vec![Complex64::new(0.6, 0.0), Complex64::new(-0.8, 0.0)],
    )
    .unwrap();
    let state =
        TreeTN::<TensorDynLen, &'static str>::from_tensors(vec![state_tensor], vec!["site0"])
            .unwrap();

    let input = DynIndex::new_dyn(2);
    let output = DynIndex::new_dyn(2);
    let op_tensor = TensorDynLen::from_dense(
        vec![output, input],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let operator =
        TreeTN::<TensorDynLen, &'static str>::from_tensors(vec![op_tensor], vec!["site0"]).unwrap();

    let result = tdvp_with_treetn_operator(
        &operator,
        state,
        &"site0",
        TdvpOptions::default().with_nsite(1),
    )
    .unwrap();

    assert_eq!(result.sweeps_completed, 1);
    assert_eq!(result.local_updates, 2);
}
