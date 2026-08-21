use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TensorElement};
use tensor4all_treetn::{fit_sum, FitOptions, TreeTN};

fn one_node<T: TensorElement>(site: &DynIndex, values: Vec<T>) -> TreeTN<IdxTensor, usize> {
    TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
        vec![0],
    )
    .unwrap()
}

#[test]
fn fit_sum_one_node_sums_targets_exactly() {
    let site = DynIndex::new_dyn(2);
    let targets = [
        one_node(&site, vec![1.0, 2.0]),
        one_node(&site, vec![3.0, 4.0]),
    ];
    let initial = one_node(&site, vec![0.0, 0.0]);
    let result = fit_sum(&targets, &initial, &0, FitOptions::new(1)).unwrap();

    assert_eq!(
        result.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![4.0, 6.0]
    );
}

#[test]
fn fit_sum_two_node_matches_dense_sum() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);
    let b2 = DynIndex::new_dyn(3);
    let b_initial = DynIndex::new_dyn(2);

    let target_one = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![s0.clone(), b1.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            IdxTensor::from_dense(vec![b1, s1.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let target_two = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), b2.clone()],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![b2, s1.clone()], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
                .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let initial = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), b_initial.clone()],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![b_initial, s1], vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();

    let expected_one = target_one.contract_to_tensor().unwrap();
    let expected_two = target_two.contract_to_tensor().unwrap();
    let expected = expected_one
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &expected_two,
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap();
    let result = fit_sum(
        &[target_one, target_two],
        &initial,
        &0,
        FitOptions::new(1).with_max_bond_dim(2),
    )
    .unwrap();
    let actual = result.contract_to_tensor().unwrap();
    assert!(actual.distance(&expected).unwrap() < 1e-10);
}

#[test]
fn fit_sum_one_node_supports_complex_dense_values() {
    use num_complex::Complex64;

    let site = DynIndex::new_dyn(2);
    let targets = [
        one_node(
            &site,
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 1.0)],
        ),
        one_node(
            &site,
            vec![Complex64::new(4.0, -1.0), Complex64::new(2.0, 5.0)],
        ),
    ];
    let result = fit_sum(
        &targets,
        &one_node(
            &site,
            vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        ),
        &0,
        FitOptions::new(1),
    )
    .unwrap();
    assert_eq!(
        result.to_dense().unwrap().to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(5.0, 1.0), Complex64::new(-1.0, 6.0)]
    );
}

#[test]
fn fit_sum_preserves_backend_real_to_complex_promotion() {
    use num_complex::Complex64;

    let site = DynIndex::new_dyn(2);
    let real = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let complex = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(
            vec![site.clone()],
            vec![Complex64::new(3.0, 4.0), Complex64::new(-1.0, 2.0)],
        )
        .unwrap()],
        vec![0],
    )
    .unwrap();
    let initial = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![0.0, 0.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let result = fit_sum(&[real, complex], &initial, &0, FitOptions::new(1)).unwrap();
    assert_eq!(
        result.to_dense().unwrap().to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(4.0, 4.0), Complex64::new(1.0, 2.0)]
    );
}

#[test]
fn fit_sum_two_node_complex_matches_dense_sum() {
    use num_complex::Complex64;

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);
    let b2 = DynIndex::new_dyn(2);
    let b_initial = DynIndex::new_dyn(2);
    let c = Complex64::new;

    let target_one = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), b1.clone()],
                vec![c(1.0, 1.0), c(0.0, 0.0), c(0.0, 0.0), c(2.0, -1.0)],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![b1, s1.clone()],
                vec![c(1.0, 2.0), c(2.0, -1.0), c(3.0, 1.0), c(4.0, -2.0)],
            )
            .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let target_two = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), b2.clone()],
                vec![c(2.0, -1.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 3.0)],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![b2, s1.clone()],
                vec![c(2.0, 1.0), c(4.0, -2.0), c(6.0, 1.0), c(8.0, -3.0)],
            )
            .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let initial = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0, b_initial.clone()],
                vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![b_initial, s1], vec![c(1.0, 0.0); 4]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();

    let expected = target_one
        .contract_to_tensor()
        .unwrap()
        .axpby(
            tensor4all_core::AnyScalar::new_complex(1.0, 0.0),
            &target_two.contract_to_tensor().unwrap(),
            tensor4all_core::AnyScalar::new_complex(1.0, 0.0),
        )
        .unwrap();
    let result = fit_sum(
        &[target_one, target_two],
        &initial,
        &0,
        FitOptions::new(1).with_max_bond_dim(2),
    )
    .unwrap();
    let actual = result.contract_to_tensor().unwrap();
    let residual = actual.distance(&expected).unwrap();
    assert!(
        residual < 1e-8,
        "complex two-node fit residual: {residual:e}"
    );
}

#[test]
fn fit_sum_zero_sweep_returns_exact_initial_clone() {
    let site = DynIndex::new_dyn(2);
    let initial = one_node(&site, vec![7.0, -2.0]);
    let targets = [one_node(&site, vec![1.0, 2.0])];
    let result = fit_sum(&targets, &initial, &0, FitOptions::new(0)).unwrap();
    assert_eq!(
        result.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![7.0, -2.0]
    );
    assert_eq!(result.canonical_region(), initial.canonical_region());
}

#[test]
fn fit_sum_reindexes_same_id_prime_site_pair_deterministically() {
    let target_base = DynIndex::new_dyn_with_tag(2, "site").unwrap();
    let target_prime = target_base.prime();
    let initial_base = DynIndex::new_dyn_with_tag(2, "site").unwrap();
    let initial_prime = initial_base.prime();
    let target = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(
            vec![target_prime.clone(), target_base.clone()],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap()],
        vec![0],
    )
    .unwrap();
    let initial = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(
            vec![initial_base.clone(), initial_prime.clone()],
            vec![0.0; 4],
        )
        .unwrap()],
        vec![0],
    )
    .unwrap();
    let expected =
        IdxTensor::from_dense(vec![initial_prime, initial_base], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = fit_sum(&[target], &initial, &0, FitOptions::new(1)).unwrap();
    assert!(result
        .to_dense()
        .unwrap()
        .isapprox(&expected, 1e-12, 0.0)
        .unwrap());
}

#[test]
fn fit_sum_rejects_validation_errors_before_shortcuts() {
    let site = DynIndex::new_dyn(2);
    let initial = one_node(&site, vec![1.0, 2.0]);
    let target = one_node(&site, vec![3.0, 4.0]);

    assert!(fit_sum(&[], &initial, &0, FitOptions::new(0)).is_err());
    assert!(fit_sum(
        std::slice::from_ref(&target),
        &TreeTN::new(),
        &0,
        FitOptions::new(0)
    )
    .is_err());
    assert!(fit_sum(&[TreeTN::new()], &initial, &0, FitOptions::new(0)).is_err());
    let missing_center_error = fit_sum(
        std::slice::from_ref(&target),
        &initial,
        &1,
        FitOptions::new(0),
    )
    .unwrap_err();
    assert!(missing_center_error.to_string().contains("fit_sum"));
    let invalid_bond_error = fit_sum(
        std::slice::from_ref(&target),
        &initial,
        &0,
        FitOptions::new(0).with_max_bond_dim(0),
    )
    .unwrap_err();
    assert!(invalid_bond_error.to_string().contains("fit_sum"));
    assert!(fit_sum(
        std::slice::from_ref(&target),
        &initial,
        &0,
        FitOptions::new(0).with_convergence_tol(f64::NAN)
    )
    .is_err());
    assert!(fit_sum(
        &[target],
        &initial,
        &0,
        FitOptions::new(0).with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(-1.0))
    )
    .is_err());
}

#[test]
fn fit_sum_rejects_topology_and_site_space_mismatches() {
    let site = DynIndex::new_dyn(2);
    let initial = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let different_topology = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0, 2.0]).unwrap()],
        vec![1],
    )
    .unwrap();
    assert!(fit_sum(&[different_topology], &initial, &0, FitOptions::new(0)).is_err());
    let wrong_dimension = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(3)], vec![1.0; 3]).unwrap()],
        vec![0],
    )
    .unwrap();
    assert!(fit_sum(&[wrong_dimension], &initial, &0, FitOptions::new(0)).is_err());
    assert!(fit_sum(
        std::slice::from_ref(&initial),
        &initial,
        &0,
        FitOptions::new(0).with_qr_rtol(1.0e-8)
    )
    .is_err());

    let bond = DynIndex::new_dyn(1);
    let extra = DynIndex::new_dyn(2);
    let extra_bond = DynIndex::new_dyn(1);
    let initial_chain = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![site.clone(), bond.clone()], vec![1.0, 0.0]).unwrap(),
            IdxTensor::from_dense(vec![bond, site.clone()], vec![1.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let extra_site_target = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![site.clone(), extra.clone(), extra_bond.clone()],
                vec![1.0; 4],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![extra_bond, site], vec![1.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    assert!(fit_sum(&[extra_site_target], &initial_chain, &0, FitOptions::new(0)).is_err());
}

#[test]
fn fit_sum_handles_branched_tree_with_site_less_internal_node() {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let target_left_bond = DynIndex::new_dyn(2);
    let target_right_bond = DynIndex::new_dyn(2);
    let other_left_bond = DynIndex::new_dyn(3);
    let other_right_bond = DynIndex::new_dyn(3);
    let initial_left_bond = DynIndex::new_dyn(2);
    let initial_right_bond = DynIndex::new_dyn(2);

    let target_one = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![left_site.clone(), target_left_bond.clone()],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![target_left_bond.clone(), target_right_bond.clone()],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![right_site.clone(), target_right_bond.clone()],
                vec![2.0, 1.0, 3.0, 4.0],
            )
            .unwrap(),
        ],
        vec![1, 0, 2],
    )
    .unwrap();
    let target_two = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![left_site.clone(), other_left_bond.clone()],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![other_left_bond.clone(), other_right_bond.clone()],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![right_site.clone(), other_right_bond.clone()],
                vec![2.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            )
            .unwrap(),
        ],
        vec![1, 0, 2],
    )
    .unwrap();
    let initial = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![left_site.clone(), initial_left_bond.clone()],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![initial_left_bond.clone(), initial_right_bond.clone()],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![right_site.clone(), initial_right_bond.clone()],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        ],
        vec![1, 0, 2],
    )
    .unwrap();

    let target_edge = target_one.edge_between(&0, &1).unwrap();
    let initial_edge = initial.edge_between(&0, &1).unwrap();
    assert_ne!(
        target_one.bond_index(target_edge),
        initial.bond_index(initial_edge),
        "target and variational cut bonds must remain distinct"
    );
    assert!(target_one.site_space(&0).unwrap().is_empty());
    assert!(initial.site_space(&0).unwrap().is_empty());

    let expected = target_one
        .contract_to_tensor()
        .unwrap()
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &target_two.contract_to_tensor().unwrap(),
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap();
    let result = fit_sum(
        &[target_one, target_two],
        &initial,
        &0,
        FitOptions::new(2).with_max_bond_dim(2),
    )
    .unwrap();
    assert!(result.site_space(&0).unwrap().is_empty());
    assert!(
        result
            .contract_to_tensor()
            .unwrap()
            .distance(&expected)
            .unwrap()
            < 1e-9
    );
}

#[test]
fn fit_sum_fixed_rank_reduces_dense_objective_without_growing_bonds() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let target_bond = DynIndex::new_dyn(2);
    let initial_bond = DynIndex::new_dyn(1);
    let target = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), target_bond.clone()],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![target_bond, s1.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let initial = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![s0, initial_bond.clone()], vec![1.0, 1.0]).unwrap(),
            IdxTensor::from_dense(vec![initial_bond, s1], vec![1.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let expected = target.contract_to_tensor().unwrap();
    let before = expected
        .sub(&initial.contract_to_tensor().unwrap())
        .unwrap()
        .norm()
        .unwrap();
    let result = fit_sum(
        &[target],
        &initial,
        &0,
        FitOptions::new(2).with_max_bond_dim(1),
    )
    .unwrap();
    let after = expected
        .sub(&result.contract_to_tensor().unwrap())
        .unwrap()
        .norm()
        .unwrap();
    assert!(
        after < before,
        "objective did not improve: before={before}, after={after}"
    );
    for (left, right) in result.site_index_network().edges() {
        let edge = result.edge_between(&left, &right).unwrap();
        assert!(result.bond_index(edge).unwrap().dim() <= 1);
    }
}

#[test]
fn fit_sum_long_chain_keeps_bonds_bounded_without_dense_reference() {
    let nsites = 30;
    let sites: Vec<_> = (0..nsites).map(|_| DynIndex::new_dyn(2)).collect();

    let make_chain = |value: f64, bond_seed: usize| {
        let mut tensors = Vec::with_capacity(nsites);
        let mut left_bond: Option<DynIndex> = None;
        for (position, site) in sites.iter().enumerate() {
            let right_bond = if position + 1 < nsites {
                Some(
                    DynIndex::new_dyn_with_tag(1, &format!("fit-sum-{bond_seed}-{position}"))
                        .unwrap(),
                )
            } else {
                None
            };
            let mut indices = Vec::new();
            if let Some(left_bond) = &left_bond {
                indices.push(left_bond.clone());
            }
            indices.push(site.clone());
            if let Some(right_bond) = &right_bond {
                indices.push(right_bond.clone());
            }
            tensors.push(IdxTensor::from_dense(indices, vec![value; 2]).unwrap());
            left_bond = right_bond;
        }
        TreeTN::<IdxTensor, usize>::from_tensors(tensors, (0..nsites).collect()).unwrap()
    };

    let target_one = make_chain(1.0, 1);
    let target_two = make_chain(2.0, 2);
    let initial = make_chain(1.0, 3);
    let result = fit_sum(
        &[target_one, target_two],
        &initial,
        &(nsites / 2),
        FitOptions::new(1).with_max_bond_dim(1),
    )
    .unwrap();
    assert_eq!(result.node_count(), nsites);
    for (left, right) in result.site_index_network().edges() {
        let edge = result.edge_between(&left, &right).unwrap();
        assert!(result.bond_index(edge).unwrap().dim() <= 1);
    }
}
