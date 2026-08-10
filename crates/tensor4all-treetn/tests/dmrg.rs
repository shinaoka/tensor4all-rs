use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, FactorizeOptions, TensorDynLen};
use tensor4all_treetn::{
    dmrg, dmrg_with_treetn_operator, DmrgError, DmrgOptions, IndexMapping, LinearOperator, TreeTN,
    TreeTopology,
};

fn col_major_offset(coords: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        offset += coord * stride;
        stride *= dim;
    }
    offset
}

fn two_site_state(values: &[f64]) -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 2]) {
    assert_eq!(values.len(), 4);
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);

    let mut left = vec![0.0; 4];
    left[col_major_offset(&[0, 0], &[2, 2])] = 1.0;
    left[col_major_offset(&[1, 1], &[2, 2])] = 1.0;

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], left).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond.clone(), s1.clone()], values.to_vec()).unwrap();
    let mut state = TreeTN::<TensorDynLen, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    state.connect(n0, &bond, n1, &bond).unwrap();
    (state, [s0, s1])
}

fn diagonal_two_site_operator(
    state_sites: &[DynIndex; 2],
    energies: [f64; 4],
) -> LinearOperator<TensorDynLen, &'static str> {
    let s0_out = DynIndex::new_dyn(2);
    let s0_in = DynIndex::new_dyn(2);
    let s1_out = DynIndex::new_dyn(2);
    let s1_in = DynIndex::new_dyn(2);

    let dims = [2, 2, 2, 2];
    let mut data = vec![0.0; dims.iter().product()];
    for s0 in 0..2 {
        for s1 in 0..2 {
            let energy = energies[s0 + 2 * s1];
            data[col_major_offset(&[s0, s0, s1, s1], &dims)] = energy;
        }
    }

    let dense = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), s1_out.clone(), s1_in.clone()],
        data,
    )
    .unwrap();

    let topology = TreeTopology::new(
        HashMap::from([
            ("site0", vec![s0_out.clone(), s0_in.clone()]),
            ("site1", vec![s1_out.clone(), s1_in.clone()]),
        ]),
        vec![("site0", "site1")],
    );
    let mpo = tensor4all_treetn::factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        &"site0",
    )
    .unwrap();

    let input_mapping = HashMap::from([
        (
            "site0",
            IndexMapping {
                true_index: state_sites[0].clone(),
                internal_index: s0_in,
            },
        ),
        (
            "site1",
            IndexMapping {
                true_index: state_sites[1].clone(),
                internal_index: s1_in,
            },
        ),
    ]);
    let output_mapping = HashMap::from([
        (
            "site0",
            IndexMapping {
                true_index: state_sites[0].clone(),
                internal_index: s0_out,
            },
        ),
        (
            "site1",
            IndexMapping {
                true_index: state_sites[1].clone(),
                internal_index: s1_out,
            },
        ),
    ]);

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn single_site_state() -> (TreeTN<TensorDynLen, &'static str>, DynIndex) {
    let site = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![site.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<TensorDynLen, &'static str>::from_tensors(vec![tensor], vec!["site0"]).unwrap();
    (state, site)
}

fn single_site_identity_operator(
    state_site: &DynIndex,
) -> LinearOperator<TensorDynLen, &'static str> {
    let out = DynIndex::new_dyn(2);
    let input = DynIndex::new_dyn(2);
    let tensor =
        TensorDynLen::from_dense(vec![out.clone(), input.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let mpo =
        TreeTN::<TensorDynLen, &'static str>::from_tensors(vec![tensor], vec!["site0"]).unwrap();
    LinearOperator::new(
        mpo,
        HashMap::from([(
            "site0",
            IndexMapping {
                true_index: state_site.clone(),
                internal_index: input,
            },
        )]),
        HashMap::from([(
            "site0",
            IndexMapping {
                true_index: state_site.clone(),
                internal_index: out,
            },
        )]),
    )
}

fn complex_two_site_operator(
    state_sites: &[DynIndex; 2],
    entries: &[(usize, usize, usize, usize, Complex64)],
) -> LinearOperator<TensorDynLen, &'static str> {
    let s0_out = DynIndex::new_dyn(2);
    let s0_in = DynIndex::new_dyn(2);
    let s1_out = DynIndex::new_dyn(2);
    let s1_in = DynIndex::new_dyn(2);

    let dims = [2, 2, 2, 2];
    let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
    for &(out0, in0, out1, in1, value) in entries {
        data[col_major_offset(&[out0, in0, out1, in1], &dims)] = value;
    }

    let dense = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), s1_out.clone(), s1_in.clone()],
        data,
    )
    .unwrap();

    let topology = TreeTopology::new(
        HashMap::from([
            ("site0", vec![s0_out.clone(), s0_in.clone()]),
            ("site1", vec![s1_out.clone(), s1_in.clone()]),
        ]),
        vec![("site0", "site1")],
    );
    let mpo = tensor4all_treetn::factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        &"site0",
    )
    .unwrap();

    let input_mapping = HashMap::from([
        (
            "site0",
            IndexMapping {
                true_index: state_sites[0].clone(),
                internal_index: s0_in,
            },
        ),
        (
            "site1",
            IndexMapping {
                true_index: state_sites[1].clone(),
                internal_index: s1_in,
            },
        ),
    ]);
    let output_mapping = HashMap::from([
        (
            "site0",
            IndexMapping {
                true_index: state_sites[0].clone(),
                internal_index: s0_out,
            },
        ),
        (
            "site1",
            IndexMapping {
                true_index: state_sites[1].clone(),
                internal_index: s1_out,
            },
        ),
    ]);

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn heisenberg_dense_operator(
    state_sites: &[DynIndex],
    edges: &[(usize, usize)],
    topology_edges: &[(&'static str, &'static str)],
) -> LinearOperator<TensorDynLen, &'static str> {
    let n_sites = state_sites.len();
    let outputs: Vec<_> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();
    let inputs: Vec<_> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();
    let mut indices = Vec::with_capacity(2 * n_sites);
    for i in 0..n_sites {
        indices.push(outputs[i].clone());
        indices.push(inputs[i].clone());
    }

    let dims = vec![2; 2 * n_sites];
    let basis_dim = 1usize << n_sites;
    let mut data = vec![0.0; dims.iter().product()];
    for input_state in 0..basis_dim {
        let bits: Vec<_> = (0..n_sites).map(|site| (input_state >> site) & 1).collect();
        for &(left, right) in edges {
            let z_left = if bits[left] == 0 { 1.0 } else { -1.0 };
            let z_right = if bits[right] == 0 { 1.0 } else { -1.0 };

            let mut coords = Vec::with_capacity(2 * n_sites);
            for &bit in &bits {
                coords.push(bit);
                coords.push(bit);
            }
            data[col_major_offset(&coords, &dims)] += z_left * z_right;

            let mut out_bits = bits.clone();
            out_bits[left] ^= 1;
            out_bits[right] ^= 1;
            let yy_coeff = if bits[left] == bits[right] { -1.0 } else { 1.0 };
            let flip_coeff = 1.0 + yy_coeff;
            if flip_coeff != 0.0 {
                let mut coords = Vec::with_capacity(2 * n_sites);
                for site in 0..n_sites {
                    coords.push(out_bits[site]);
                    coords.push(bits[site]);
                }
                data[col_major_offset(&coords, &dims)] += flip_coeff;
            }
        }
    }

    let dense = TensorDynLen::from_dense(indices, data).unwrap();
    let topology = TreeTopology::new(
        (0..n_sites)
            .map(|i| (site_name(i), vec![outputs[i].clone(), inputs[i].clone()]))
            .collect(),
        topology_edges.to_vec(),
    );
    let mpo = tensor4all_treetn::factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        &"site0",
    )
    .unwrap();

    let input_mapping = (0..n_sites)
        .map(|i| {
            (
                site_name(i),
                IndexMapping {
                    true_index: state_sites[i].clone(),
                    internal_index: inputs[i].clone(),
                },
            )
        })
        .collect();
    let output_mapping = (0..n_sites)
        .map(|i| {
            (
                site_name(i),
                IndexMapping {
                    true_index: state_sites[i].clone(),
                    internal_index: outputs[i].clone(),
                },
            )
        })
        .collect();

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn site_name(i: usize) -> &'static str {
    match i {
        0 => "site0",
        1 => "site1",
        2 => "site2",
        3 => "site3",
        _ => panic!("test helper supports at most four sites"),
    }
}

fn three_site_star_plus_state() -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 3]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let dense =
        TensorDynLen::from_dense(vec![s0.clone(), s1.clone(), s2.clone()], vec![1.0; 8]).unwrap();
    let topology = TreeTopology::new(
        HashMap::from([
            ("site0", vec![s0.clone()]),
            ("site1", vec![s1.clone()]),
            ("site2", vec![s2.clone()]),
        ]),
        vec![("site0", "site1"), ("site0", "site2")],
    );
    let state = tensor4all_treetn::factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        &"site0",
    )
    .unwrap();
    (state, [s0, s1, s2])
}

fn four_site_star_asymmetric_state() -> (TreeTN<TensorDynLen, &'static str>, [DynIndex; 4]) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let dense = TensorDynLen::from_dense(
        vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()],
        vec![
            0.7, -0.3, 1.1, 0.2, -0.8, 0.5, 0.9, -1.2, 0.4, 1.3, -0.6, 0.1, 1.0, -0.9, 0.3, 0.8,
        ],
    )
    .unwrap();
    let topology = TreeTopology::new(
        HashMap::from([
            ("site0", vec![s0.clone()]),
            ("site1", vec![s1.clone()]),
            ("site2", vec![s2.clone()]),
            ("site3", vec![s3.clone()]),
        ]),
        vec![("site0", "site1"), ("site0", "site2"), ("site0", "site3")],
    );
    let state = tensor4all_treetn::factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        &"site0",
    )
    .unwrap();
    (state, [s0, s1, s2, s3])
}

fn star_sum_z_operator(state_sites: &[DynIndex; 3]) -> LinearOperator<TensorDynLen, &'static str> {
    let out0 = DynIndex::new_dyn(2);
    let in0 = DynIndex::new_dyn(2);
    let out1 = DynIndex::new_dyn(2);
    let in1 = DynIndex::new_dyn(2);
    let out2 = DynIndex::new_dyn(2);
    let in2 = DynIndex::new_dyn(2);

    let dims = [2, 2, 2, 2, 2, 2];
    let mut data = vec![0.0; dims.iter().product()];
    for s0 in 0..2 {
        for s1 in 0..2 {
            for s2 in 0..2 {
                let z0 = if s0 == 0 { 1.0 } else { -1.0 };
                let z1 = if s1 == 0 { 1.0 } else { -1.0 };
                let z2 = if s2 == 0 { 1.0 } else { -1.0 };
                data[col_major_offset(&[s0, s0, s1, s1, s2, s2], &dims)] = z0 + z1 + z2;
            }
        }
    }
    let dense = TensorDynLen::from_dense(
        vec![
            out0.clone(),
            in0.clone(),
            out1.clone(),
            in1.clone(),
            out2.clone(),
            in2.clone(),
        ],
        data,
    )
    .unwrap();
    let topology = TreeTopology::new(
        HashMap::from([
            ("site0", vec![out0.clone(), in0.clone()]),
            ("site1", vec![out1.clone(), in1.clone()]),
            ("site2", vec![out2.clone(), in2.clone()]),
        ]),
        vec![("site0", "site1"), ("site0", "site2")],
    );
    let mpo = tensor4all_treetn::factorize_tensor_to_treetn_with(
        &dense,
        &topology,
        FactorizeOptions::svd(),
        &"site0",
    )
    .unwrap();

    let input_mapping = HashMap::from([
        (
            "site0",
            IndexMapping {
                true_index: state_sites[0].clone(),
                internal_index: in0,
            },
        ),
        (
            "site1",
            IndexMapping {
                true_index: state_sites[1].clone(),
                internal_index: in1,
            },
        ),
        (
            "site2",
            IndexMapping {
                true_index: state_sites[2].clone(),
                internal_index: in2,
            },
        ),
    ]);
    let output_mapping = HashMap::from([
        (
            "site0",
            IndexMapping {
                true_index: state_sites[0].clone(),
                internal_index: out0,
            },
        ),
        (
            "site1",
            IndexMapping {
                true_index: state_sites[1].clone(),
                internal_index: out1,
            },
        ),
        (
            "site2",
            IndexMapping {
                true_index: state_sites[2].clone(),
                internal_index: out2,
            },
        ),
    ]);

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

#[test]
fn dmrg_rejects_one_site_option() {
    let (state, sites) = two_site_state(&[0.3, 0.4, 0.5, 0.6]);
    let operator = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);
    let err = dmrg(
        &operator,
        state,
        &"site0",
        DmrgOptions::default().with_nsite(1),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        DmrgError::UnsupportedNsite {
            requested: 1,
            supported: 2
        }
    ));
}

#[test]
fn dmrg_rejects_invalid_options() {
    let (state, sites) = two_site_state(&[0.3, 0.4, 0.5, 0.6]);
    let operator = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);

    let zero_sweeps = DmrgOptions {
        nsweeps: 0,
        ..Default::default()
    };
    let err = dmrg(&operator, state.clone(), &"site0", zero_sweeps).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::InvalidOption {
            option: "nsweeps",
            ..
        }
    ));

    let zero_max_bond_dim = DmrgOptions {
        max_bond_dim: Some(0),
        ..Default::default()
    };
    let err = dmrg(&operator, state.clone(), &"site0", zero_max_bond_dim).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::InvalidOption {
            option: "max_bond_dim",
            ..
        }
    ));

    let negative_energy_tol = DmrgOptions {
        energy_tol: Some(-1.0),
        ..Default::default()
    };
    let err = dmrg(&operator, state.clone(), &"site0", negative_energy_tol).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::InvalidOption {
            option: "energy_tol",
            ..
        }
    ));

    let negative_real_scalar_tol = DmrgOptions {
        real_scalar_tol: -1.0,
        ..Default::default()
    };
    let err = dmrg(&operator, state, &"site0", negative_real_scalar_tol).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::InvalidOption {
            option: "real_scalar_tol",
            ..
        }
    ));
}

#[test]
fn dmrg_rejects_single_node_two_site_sweep() {
    let (state, site) = single_site_state();
    let operator = single_site_identity_operator(&site);

    let err = dmrg(&operator, state, &"site0", DmrgOptions::default()).unwrap_err();
    assert!(matches!(err, DmrgError::EmptyTwoSiteSweep));
}

#[test]
fn dmrg_rejects_missing_mapping() {
    let (state, sites) = two_site_state(&[0.3, 0.4, 0.5, 0.6]);
    let operator = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);
    let mut input_mapping = operator.input_mappings().clone();
    input_mapping.remove("site1");
    let operator = LinearOperator::new_multi(
        operator.mpo().clone(),
        input_mapping,
        operator.output_mappings().clone(),
    );

    let err = dmrg(&operator, state, &"site0", DmrgOptions::default()).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::MissingMapping {
            node,
            role: "input"
        } if node == "\"site1\""
    ));
}

#[test]
fn dmrg_rejects_mapping_with_same_dimension_but_different_true_index() {
    let (state, sites) = two_site_state(&[0.3, 0.4, 0.5, 0.6]);
    let operator = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);
    let mut input_mapping = operator.input_mappings().clone();
    input_mapping.get_mut("site0").unwrap()[0].true_index = DynIndex::new_dyn(2);
    let operator = LinearOperator::new_multi(
        operator.mpo().clone(),
        input_mapping,
        operator.output_mappings().clone(),
    );

    let err = dmrg(&operator, state, &"site0", DmrgOptions::default()).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::InvalidMapping {
            node,
            reason
        } if node == "\"site0\"" && reason.contains("input true index")
    ));
}

#[test]
fn dmrg_rejects_mapping_dimension_mismatch() {
    let (state, sites) = two_site_state(&[0.3, 0.4, 0.5, 0.6]);
    let operator = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);
    let mut input_mapping = operator.input_mappings().clone();
    input_mapping.get_mut("site0").unwrap()[0].internal_index = DynIndex::new_dyn(3);
    let operator = LinearOperator::new_multi(
        operator.mpo().clone(),
        input_mapping,
        operator.output_mappings().clone(),
    );

    let err = dmrg(&operator, state, &"site0", DmrgOptions::default()).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::InvalidMapping {
            node,
            reason
        } if node == "\"site0\"" && reason.contains("dimensions")
    ));
}

#[test]
fn dmrg_two_site_diagonal_hamiltonian_finds_lowest_energy() {
    let (state, sites) = two_site_state(&[0.2, 0.3, 0.4, 0.5]);
    let operator = diagonal_two_site_operator(&sites, [2.0, -1.5, 0.5, -3.0]);

    let result = dmrg(
        &operator,
        state,
        &"site0",
        DmrgOptions::default()
            .with_nsweeps(2)
            .with_energy_tol(1.0e-12),
    )
    .unwrap();

    assert!(
        (result.energy + 3.0).abs() < 1.0e-10,
        "energy={}",
        result.energy
    );
    assert!(result.local_updates >= 1);
    assert!(result.max_residual_norm < 1.0e-8);
}

#[test]
fn dmrg_two_site_heisenberg_singlet_builds_entangled_bond() {
    let (state, sites) = two_site_state(&[0.7, -0.2, 0.5, 1.1]);
    let operator = heisenberg_dense_operator(&sites, &[(0, 1)], &[("site0", "site1")]);

    let result = dmrg(
        &operator,
        state,
        &"site0",
        DmrgOptions::default()
            .with_nsweeps(4)
            .with_max_bond_dim(4)
            .with_energy_tol(1.0e-12),
    )
    .unwrap();

    assert!(
        (result.energy + 3.0).abs() < 1.0e-10,
        "energy={}",
        result.energy
    );
    assert!(result.state.link_dims().iter().any(|&dim| dim >= 2));
    assert!(result.max_residual_norm < 1.0e-8);
}

#[test]
fn dmrg_rejects_multiple_site_mappings_per_node() {
    let (state, sites) = two_site_state(&[0.2, 0.3, 0.4, 0.5]);
    let operator = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);

    let extra_true = DynIndex::new_dyn(2);
    let extra_internal = DynIndex::new_dyn(2);
    let mut input_mapping = operator.input_mappings().clone();
    input_mapping.get_mut("site0").unwrap().push(IndexMapping {
        true_index: extra_true.clone(),
        internal_index: extra_internal.clone(),
    });
    let output_mapping = operator.output_mappings().clone();
    let operator = LinearOperator::new_multi(operator.mpo().clone(), input_mapping, output_mapping);

    let err = dmrg(&operator, state, &"site0", DmrgOptions::default()).unwrap_err();
    assert!(matches!(
        err,
        DmrgError::UnsupportedMultipleSiteMappings {
            node,
            role: "input",
            count: 2
        } if node == "\"site0\""
    ));
}

#[test]
fn dmrg_two_site_complex_hermitian_operator_keeps_real_energy() {
    let (state, sites) = two_site_state(&[0.2, 0.3, 0.4, 0.5]);
    let mut entries = Vec::new();
    for s1 in 0..2 {
        entries.push((0, 1, s1, s1, Complex64::new(0.0, -1.0)));
        entries.push((1, 0, s1, s1, Complex64::new(0.0, 1.0)));
    }
    let operator = complex_two_site_operator(&sites, &entries);

    let result = dmrg(
        &operator,
        state,
        &"site0",
        DmrgOptions::default().with_nsweeps(2),
    )
    .unwrap();

    assert!((result.energy + 1.0).abs() < 1.0e-10);
}

#[test]
fn dmrg_non_chain_star_sum_z_finds_lowest_energy() {
    let (state, sites) = three_site_star_plus_state();
    let operator = star_sum_z_operator(&sites);

    let result = dmrg(
        &operator,
        state,
        &"site0",
        DmrgOptions::default()
            .with_nsweeps(3)
            .with_max_bond_dim(8)
            .with_energy_tol(1.0e-12),
    )
    .unwrap();

    assert!((result.energy + 3.0).abs() < 1.0e-10);
}

#[test]
fn dmrg_branching_star_heisenberg_finds_entangled_ground_state() {
    let (state, sites) = four_site_star_asymmetric_state();
    let operator = heisenberg_dense_operator(
        &sites,
        &[(0, 1), (0, 2), (0, 3)],
        &[("site0", "site1"), ("site0", "site2"), ("site0", "site3")],
    );

    let result = dmrg(
        &operator,
        state,
        &"site0",
        DmrgOptions::default()
            .with_nsweeps(5)
            .with_max_bond_dim(8)
            .with_energy_tol(1.0e-12),
    )
    .unwrap();

    assert!(
        (result.energy + 5.0).abs() < 1.0e-9,
        "energy={}",
        result.energy
    );
    assert!(result.state.link_dims().iter().any(|&dim| dim >= 2));
}

#[test]
fn dmrg_rejects_non_hermitian_projected_operator() {
    let (state, sites) = two_site_state(&[0.2, 0.3, 0.4, 0.5]);
    let mut entries = Vec::new();
    for s1 in 0..2 {
        entries.push((0, 1, s1, s1, Complex64::new(1.0, 0.0)));
    }
    let operator = complex_two_site_operator(&sites, &entries);

    let err = dmrg(
        &operator,
        state,
        &"site0",
        DmrgOptions::default().with_nsweeps(1),
    )
    .unwrap_err();
    let err_text = err.to_string();
    assert!(
        err_text.contains("DMRG sweep failed") && err_text.contains("Hermitian"),
        "{err_text}"
    );
}

#[test]
fn dmrg_rejects_missing_center_node() {
    let (state, sites) = two_site_state(&[0.3, 0.4, 0.5, 0.6]);
    let operator = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);
    let err = dmrg(&operator, state, &"missing", DmrgOptions::default()).unwrap_err();
    assert!(matches!(err, DmrgError::MissingCenter { .. }));
}

#[test]
fn dmrg_rejects_topology_mismatch_between_state_and_operator() {
    let (state, sites) = two_site_state(&[0.3, 0.4, 0.5, 0.6]);
    let operator = single_site_identity_operator(&sites[0]);
    let err = dmrg(&operator, state, &"site0", DmrgOptions::default()).unwrap_err();
    assert!(matches!(err, DmrgError::TopologyMismatch));
}

#[test]
fn dmrg_rejects_single_node_two_site_sweep_path() {
    let (state, site) = single_site_state();
    let operator = single_site_identity_operator(&site);
    let err = dmrg(&operator, state, &"site0", DmrgOptions::default()).unwrap_err();
    assert!(matches!(err, DmrgError::EmptyTwoSiteSweep));
}

#[test]
fn dmrg_with_treetn_operator_solves_ground_state() {
    let (state, sites) = two_site_state(&[0.0, 0.0, 0.0, 1.0]);
    let linop = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);
    let mpo = linop.mpo().clone();
    let result = dmrg_with_treetn_operator(&mpo, state, &"site0", DmrgOptions::default()).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn dmrg_with_treetn_operator_rejects_missing_center() {
    let (state, sites) = two_site_state(&[0.0, 0.0, 0.0, 1.0]);
    let linop = diagonal_two_site_operator(&sites, [1.0, 2.0, 3.0, 4.0]);
    let mpo = linop.mpo().clone();
    let err =
        dmrg_with_treetn_operator(&mpo, state, &"missing", DmrgOptions::default()).unwrap_err();
    assert!(matches!(err, DmrgError::MissingCenter { .. }));
}
