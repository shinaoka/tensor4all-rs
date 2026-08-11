use super::*;
use crate::random::{random_treetn, LinkSpace};
use crate::{
    LinearOperatorIndexBindingError, LinearOperatorTaggedApplyError, NumberedTagSelectionError,
    SiteIndexNetwork,
};
use std::collections::{HashMap, HashSet};
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::{ColMajorArrayRef, SvdTruncationPolicy, TensorDynLen};
use tensor4all_tensorbackend::StorageKind;

type DynIndex = Index<DynId, TagSet>;
type ChainStateAndIdentity = (
    TreeTN<TensorDynLen, String>,
    LinearOperator<TensorDynLen, String>,
    Vec<(String, DynIndex)>,
);

fn make_index(dim: usize) -> DynIndex {
    Index::new_dyn(dim)
}

fn build_identity_operator(sites: &[(String, DynIndex)]) -> LinearOperator<TensorDynLen, String> {
    let mut mpo = TreeTN::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for (name, true_index) in sites {
        let internal_input = make_index(true_index.dim());
        let internal_output = make_index(true_index.dim());
        let tensor = TensorDynLen::from_dense(
            vec![internal_output.clone(), internal_input.clone()],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        mpo.add_tensor(name.clone(), tensor).unwrap();
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_input,
            },
        );
        output_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_output,
            },
        );
    }

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn build_bonded_identity_operator(
    sites: &[(String, DynIndex)],
) -> LinearOperator<TensorDynLen, String> {
    assert_eq!(sites.len(), 2);

    let mut mpo = TreeTN::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let bond = make_index(1);

    for (name, true_index) in sites {
        let internal_input = make_index(true_index.dim());
        let internal_output = make_index(true_index.dim());
        let tensor = if name == &sites[0].0 {
            TensorDynLen::from_dense(
                vec![
                    internal_output.clone(),
                    internal_input.clone(),
                    bond.clone(),
                ],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap()
        } else {
            TensorDynLen::from_dense(
                vec![
                    bond.clone(),
                    internal_output.clone(),
                    internal_input.clone(),
                ],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap()
        };

        mpo.add_tensor(name.clone(), tensor).unwrap();
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_input,
            },
        );
        output_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_output,
            },
        );
    }

    let n0 = mpo.node_index(&sites[0].0).unwrap();
    let n1 = mpo.node_index(&sites[1].0).unwrap();
    mpo.connect(n0, &bond, n1, &bond).unwrap();

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn build_redundant_bonded_identity_operator(
    sites: &[(String, DynIndex)],
    mpo_bond_dim: usize,
) -> LinearOperator<TensorDynLen, String> {
    assert_eq!(sites.len(), 2);

    let mut mpo = TreeTN::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let bond = make_index(mpo_bond_dim);

    for (site_pos, (name, true_index)) in sites.iter().enumerate() {
        let internal_input = make_index(true_index.dim());
        let internal_output = make_index(true_index.dim());
        let dim = true_index.dim();
        let tensor = if site_pos == 0 {
            let mut data = vec![0.0; dim * dim * mpo_bond_dim];
            for bond_value in 0..mpo_bond_dim {
                for site_value in 0..dim {
                    data[site_value + dim * (site_value + dim * bond_value)] = 1.0;
                }
            }
            TensorDynLen::from_dense(
                vec![
                    internal_output.clone(),
                    internal_input.clone(),
                    bond.clone(),
                ],
                data,
            )
            .unwrap()
        } else {
            let mut data = vec![0.0; mpo_bond_dim * dim * dim];
            for bond_value in 0..mpo_bond_dim {
                for site_value in 0..dim {
                    data[bond_value + mpo_bond_dim * (site_value + dim * site_value)] =
                        1.0 / mpo_bond_dim as f64;
                }
            }
            TensorDynLen::from_dense(
                vec![
                    bond.clone(),
                    internal_output.clone(),
                    internal_input.clone(),
                ],
                data,
            )
            .unwrap()
        };

        mpo.add_tensor(name.clone(), tensor).unwrap();
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_input,
            },
        );
        output_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_output,
            },
        );
    }

    let n0 = mpo.node_index(&sites[0].0).unwrap();
    let n1 = mpo.node_index(&sites[1].0).unwrap();
    mpo.connect(n0, &bond, n1, &bond).unwrap();

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn build_chain_state() -> (TreeTN<TensorDynLen, String>, Vec<(String, DynIndex)>) {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let s3 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);
    let b23 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        TensorDynLen::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 =
        TensorDynLen::from_dense(vec![b12.clone(), s2.clone(), b23.clone()], vec![1.0; 8]).unwrap();
    let t3 = TensorDynLen::from_dense(vec![b23.clone(), s3.clone()], vec![1.0; 4]).unwrap();

    let state = TreeTN::from_tensors(
        vec![t0, t1, t2, t3],
        vec![
            "site0".to_string(),
            "site1".to_string(),
            "site2".to_string(),
            "site3".to_string(),
        ],
    )
    .unwrap();

    (
        state,
        vec![
            ("site0".to_string(), s0),
            ("site1".to_string(), s1),
            ("site2".to_string(), s2),
            ("site3".to_string(), s3),
        ],
    )
}

#[test]
fn apply_linear_operator_supports_multiple_index_mappings_per_node() {
    let x = make_index(2);
    let y = make_index(2);
    let state_tensor =
        TensorDynLen::from_dense(vec![x.clone(), y.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let state =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![state_tensor], vec!["site".to_string()])
            .unwrap();

    let x_in = make_index(2);
    let y_in = make_index(2);
    let x_out = make_index(2);
    let y_out = make_index(2);
    let mut data = vec![0.0; 16];
    for xi in 0..2 {
        for yi in 0..2 {
            let xo = xi;
            let yo = yi;
            data[xo + 2 * (yo + 2 * (xi + 2 * yi))] = 1.0;
        }
    }
    let mpo_tensor = TensorDynLen::from_dense(
        vec![x_out.clone(), y_out.clone(), x_in.clone(), y_in.clone()],
        data,
    )
    .unwrap();
    let mpo =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![mpo_tensor], vec!["site".to_string()])
            .unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "site".to_string(),
        vec![
            IndexMapping {
                true_index: x.clone(),
                internal_index: x_in,
            },
            IndexMapping {
                true_index: y.clone(),
                internal_index: y_in,
            },
        ],
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "site".to_string(),
        vec![
            IndexMapping {
                true_index: x.clone(),
                internal_index: x_out,
            },
            IndexMapping {
                true_index: y.clone(),
                internal_index: y_out,
            },
        ],
    );

    let operator = LinearOperator::new_multi(mpo, input_mapping, output_mapping);
    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert!(
        result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap()
            < 1e-12
    );
}

#[test]
fn apply_linear_operator_to_indices_binds_explicit_external_indices() {
    let state_index = make_index(2);
    let state_tensor = TensorDynLen::from_dense(vec![state_index.clone()], vec![2.0, 5.0]).unwrap();
    let state =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![state_tensor], vec!["site".to_string()])
            .unwrap();

    let op_true_input = make_index(2);
    let op_true_output = make_index(2);
    let internal_input = make_index(2);
    let internal_output = make_index(2);
    let mpo_tensor = TensorDynLen::from_dense(
        vec![internal_output.clone(), internal_input.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let mpo =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![mpo_tensor], vec!["site".to_string()])
            .unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "site".to_string(),
        IndexMapping {
            true_index: op_true_input.clone(),
            internal_index: internal_input,
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "site".to_string(),
        IndexMapping {
            true_index: op_true_output.clone(),
            internal_index: internal_output,
        },
    );
    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    let result = apply_linear_operator_to_indices(
        &operator,
        &state,
        &[(op_true_input, state_index.clone())],
        &[(op_true_output, state_index.clone())],
        ApplyOptions::naive(),
    )
    .unwrap();

    assert!(
        result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap()
            < 1e-12
    );
}

#[test]
fn bind_linear_operator_indices_reports_typed_error_paths() {
    let op_true_input = make_index(2);
    let operator = build_identity_operator(&[("site".to_string(), op_true_input.clone())]);
    let same_id_prime = op_true_input.prime();

    let missing = bind_linear_operator_indices(&operator, &[(same_id_prime, make_index(2))], &[])
        .unwrap_err();
    assert!(matches!(
        missing,
        LinearOperatorIndexBindingError::MissingSourceIndex { role: "input", .. }
    ));

    let mismatch =
        bind_linear_operator_indices(&operator, &[(op_true_input.clone(), make_index(3))], &[])
            .unwrap_err();
    assert!(matches!(
        mismatch,
        LinearOperatorIndexBindingError::DimensionMismatch {
            role: "input",
            old_dim: 2,
            new_dim: 3,
        }
    ));

    let duplicate = bind_linear_operator_indices(
        &operator,
        &[
            (op_true_input.clone(), make_index(2)),
            (op_true_input, make_index(2)),
        ],
        &[],
    )
    .unwrap_err();
    assert!(matches!(
        duplicate,
        LinearOperatorIndexBindingError::DuplicateSourceIndex { role: "input", .. }
    ));
}

#[test]
fn apply_linear_operator_to_numbered_tags_binds_state_indices_in_tag_order() {
    let k1 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let k2 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    let bond = make_index(1);
    let state = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![
            TensorDynLen::from_dense(vec![k1.clone(), bond.clone()], vec![1.0, 2.0]).unwrap(),
            TensorDynLen::from_dense(vec![bond, k2.clone()], vec![3.0, 5.0]).unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();
    let operator = build_bonded_identity_operator(&[
        ("site0".to_string(), make_index(2)),
        ("site1".to_string(), make_index(2)),
    ]);

    let result =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap();

    assert!(
        result
            .to_dense()
            .unwrap()
            .distance(&state.to_dense().unwrap())
            .unwrap()
            < 1.0e-12
    );
}

#[test]
fn apply_linear_operator_to_numbered_tags_reports_missing_tag() {
    let k1 = Index::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let state = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![TensorDynLen::from_dense(vec![k1], vec![1.0, 2.0]).unwrap()],
        vec!["site0".to_string()],
    )
    .unwrap();
    let operator = build_identity_operator(&[
        ("site0".to_string(), make_index(2)),
        ("site1".to_string(), make_index(2)),
    ]);

    let err =
        apply_linear_operator_to_numbered_tags(&operator, &state, "k", 1, ApplyOptions::naive())
            .unwrap_err();

    assert!(matches!(
        err,
        LinearOperatorTaggedApplyError::SelectTaggedIndices(
            NumberedTagSelectionError::MissingTag { ref tag }
        ) if tag == "k=2"
    ));
}

fn build_tree_state() -> (TreeTN<TensorDynLen, String>, Vec<(String, DynIndex)>) {
    let s_a = make_index(2);
    let s_b = make_index(2);
    let s_c = make_index(2);
    let s_d = make_index(2);
    let s_e = make_index(2);
    let b_ab = make_index(2);
    let b_bc = make_index(2);
    let b_bd = make_index(2);
    let b_be = make_index(2);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone(), b_ab.clone()], vec![1.0; 4]).unwrap();
    let t_b = TensorDynLen::from_dense(
        vec![b_ab.clone(), s_b.clone(), b_bc.clone(), b_bd.clone()],
        vec![1.0; 16],
    )
    .unwrap();
    let t_c = TensorDynLen::from_dense(vec![b_bc.clone(), s_c.clone()], vec![1.0; 4]).unwrap();
    let t_d = TensorDynLen::from_dense(vec![b_bd.clone(), s_d.clone(), b_be.clone()], vec![1.0; 8])
        .unwrap();
    let t_e = TensorDynLen::from_dense(vec![b_be.clone(), s_e.clone()], vec![1.0; 4]).unwrap();

    let state = TreeTN::from_tensors(
        vec![t_a, t_b, t_c, t_d, t_e],
        vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
        ],
    )
    .unwrap();

    (
        state,
        vec![
            ("A".to_string(), s_a),
            ("B".to_string(), s_b),
            ("C".to_string(), s_c),
            ("D".to_string(), s_d),
            ("E".to_string(), s_e),
        ],
    )
}

fn build_uniform_chain_state_and_identity_operator(
    length: usize,
    state_bond_dim: usize,
    mpo_bond_dim: usize,
) -> ChainStateAndIdentity {
    let site_indices: Vec<_> = (0..length).map(|_| make_index(2)).collect();
    let state_bonds: Vec<_> = (0..length.saturating_sub(1))
        .map(|_| make_index(state_bond_dim))
        .collect();
    let mpo_bonds: Vec<_> = (0..length.saturating_sub(1))
        .map(|_| make_index(mpo_bond_dim))
        .collect();

    let mut state_tensors = Vec::with_capacity(length);
    let mut mpo_tensors = Vec::with_capacity(length);
    let mut node_names = Vec::with_capacity(length);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for site in 0..length {
        let name = format!("site{site}");
        node_names.push(name.clone());

        let mut state_indices = Vec::new();
        if site > 0 {
            state_indices.push(state_bonds[site - 1].clone());
        }
        state_indices.push(site_indices[site].clone());
        if site + 1 < length {
            state_indices.push(state_bonds[site].clone());
        }
        let state_len = state_indices.iter().map(|idx| idx.dim()).product();
        state_tensors.push(TensorDynLen::from_dense(state_indices, vec![1.0; state_len]).unwrap());

        let internal_input = make_index(2);
        let internal_output = make_index(2);
        let mut mpo_indices = Vec::new();
        if site > 0 {
            mpo_indices.push(mpo_bonds[site - 1].clone());
        }
        mpo_indices.push(internal_output.clone());
        mpo_indices.push(internal_input.clone());
        if site + 1 < length {
            mpo_indices.push(mpo_bonds[site].clone());
        }

        let mpo_len = mpo_indices.iter().map(|idx| idx.dim()).product();
        let mut mpo_data = vec![0.0; mpo_len];
        let left_dim = if site > 0 { mpo_bond_dim } else { 1 };
        let right_dim = if site + 1 < length { mpo_bond_dim } else { 1 };
        for left in 0..left_dim {
            for output in 0..2 {
                for input in 0..2 {
                    for right in 0..right_dim {
                        if output == input {
                            let offset = if site == 0 {
                                output + 2 * (input + 2 * right)
                            } else if site + 1 == length {
                                left + left_dim * (output + 2 * input)
                            } else {
                                left + left_dim * (output + 2 * (input + 2 * right))
                            };
                            mpo_data[offset] = 1.0 / mpo_bond_dim as f64;
                        }
                    }
                }
            }
        }
        mpo_tensors.push(TensorDynLen::from_dense(mpo_indices, mpo_data).unwrap());

        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: site_indices[site].clone(),
                internal_index: internal_input,
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index: site_indices[site].clone(),
                internal_index: internal_output,
            },
        );
    }

    let state = TreeTN::from_tensors(state_tensors, node_names.clone()).unwrap();
    let mpo = TreeTN::from_tensors(mpo_tensors, node_names).unwrap();
    let sites = site_indices
        .into_iter()
        .enumerate()
        .map(|(i, idx)| (format!("site{i}"), idx))
        .collect();

    (
        state,
        LinearOperator::new(mpo, input_mapping, output_mapping),
        sites,
    )
}

fn assert_identity_application(
    state: &TreeTN<TensorDynLen, String>,
    operator: &LinearOperator<TensorDynLen, String>,
) {
    use crate::operator::apply_linear_operator;

    let result = apply_linear_operator(operator, state, ApplyOptions::default()).unwrap();
    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn test_apply_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-10)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = ApplyOptions::zipup()
        .with_max_rank(50)
        .with_svd_policy(policy);
    assert_eq!(opts.method, ContractionMethod::Zipup);
    assert_eq!(opts.max_rank, Some(50));
    assert_eq!(opts.svd_policy, Some(policy));
    assert_eq!(opts.qr_rtol, None);

    let fit = ApplyOptions::fit().with_nfullsweeps(3);
    assert_eq!(fit.method, ContractionMethod::Fit);
    assert_eq!(fit.nfullsweeps, 3);

    let naive = ApplyOptions::naive();
    assert_eq!(naive.method, ContractionMethod::Naive);
}

#[test]
fn test_linear_operator_tensor_index() {
    // Create a simple LinearOperator
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    net.add_node(
        "N0".to_string(),
        [s0_in.clone(), s0_out.clone()]
            .into_iter()
            .collect::<HashSet<_>>(),
    )
    .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space).unwrap();

    let true_s0 = make_index(2);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    output_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_out.clone(),
        },
    );

    let lin_op = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test external_indices
    let ext_indices = lin_op.external_indices();
    assert_eq!(ext_indices.len(), 2);
    assert_eq!(lin_op.num_external_indices(), 2);

    // Test replaceind
    let new_idx = make_index(2);
    let replaced = lin_op.replaceind(&true_s0, &new_idx).unwrap();
    assert!(replaced.get_input_mapping(&"N0".to_string()).is_some());

    let same_id_prime = true_s0.prime();
    let wrong_prime = lin_op.replaceind(&same_id_prime, &make_index(2));
    assert!(wrong_prime.is_err());
}

#[test]
fn test_linear_operator_replaceind_errors() {
    let operator = build_identity_operator(&[("site0".to_string(), make_index(2))]);

    let mismatch = operator.replaceind(&make_index(2), &make_index(3));
    assert!(mismatch.is_err());

    let unknown = operator.replaceind(&make_index(2), &make_index(2));
    assert!(unknown.is_err());
}

#[test]
fn test_arc_linear_operator_cow() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0 = make_index(2);
    net.add_node(
        "N0".to_string(),
        [s0.clone()].into_iter().collect::<HashSet<_>>(),
    )
    .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space).unwrap();

    let arc_op = ArcLinearOperator::new(mpo, HashMap::new(), HashMap::new());

    // Clone should share the Arc
    let arc_op2 = arc_op.clone();
    assert!(Arc::ptr_eq(&arc_op.mpo, &arc_op2.mpo));

    // Mutating one should not affect the other (CoW)
    let mut arc_op3 = arc_op.clone();
    let _mpo_mut = arc_op3.mpo_mut();
    // After make_mut, the Arcs should be different if there were other refs
    // (In this case, arc_op still holds a reference)
    assert!(!Arc::ptr_eq(&arc_op.mpo, &arc_op3.mpo));
}

#[test]
fn test_arc_linear_operator_accessors_and_conversion() {
    let op = build_identity_operator(&[("site0".to_string(), make_index(2))]);
    let arc_op = ArcLinearOperator::from_linear_operator(op.clone());

    assert_eq!(arc_op.node_names(), op.node_names());
    assert!(arc_op.get_input_mapping(&"site0".to_string()).is_some());
    assert!(arc_op.get_output_mapping(&"site0".to_string()).is_some());
    assert_eq!(arc_op.input_mappings().len(), 1);
    assert_eq!(arc_op.output_mappings().len(), 1);

    let arc_clone = arc_op.clone();
    let op2 = arc_clone.into_linear_operator();
    assert_eq!(op2.mpo.node_count(), op.mpo.node_count());
}

#[test]
fn test_apply_linear_operator_full_coverage() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 2-site state
    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let b01 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();

    // Create identity operator covering both sites
    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test apply with default options
    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), 2);

    // Test apply with different methods
    let result_fit = apply_linear_operator(&operator, &state, ApplyOptions::fit()).unwrap();
    assert_eq!(result_fit.node_count(), 2);

    let result_naive = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    assert_eq!(result_naive.node_count(), 2);
}

#[test]
fn naive_apply_preserves_product_link_space_for_redundant_mpo_bond() {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let state_bond = make_index(2);
    let state = TreeTN::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![s0.clone(), state_bond.clone()],
                vec![1.0, 3.0, 2.0, 5.0],
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![state_bond.clone(), s1.clone()],
                vec![7.0, 11.0, 13.0, 17.0],
            )
            .unwrap(),
        ],
        vec!["site0".to_string(), "site1".to_string()],
    )
    .unwrap();
    let operator = build_redundant_bonded_identity_operator(
        &[
            ("site0".to_string(), s0.clone()),
            ("site1".to_string(), s1.clone()),
        ],
        3,
    );

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);
    let edge = result
        .edge_between(&"site0".to_string(), &"site1".to_string())
        .unwrap();
    assert_eq!(result.bond_index(edge).unwrap().dim(), 2 * 3);

    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn naive_apply_full_product_identity_embeds_on_state_topology() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&sites);

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert_eq!(result.node_count(), state.node_count());
    assert_eq!(result.edge_count(), state.edge_count());

    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn naive_apply_noncontiguous_bonded_identity_uses_compact_bridge_delta() {
    let (state, sites) = build_chain_state();
    let operator =
        build_redundant_bonded_identity_operator(&[sites[0].clone(), sites[2].clone()], 2);

    let extended = extend_operator_to_full_space(&operator, &state).unwrap();
    let middle = extended.mpo.node_index(&sites[1].0).unwrap();
    let middle_tensor = extended.mpo.tensor(middle).unwrap();

    assert_eq!(
        middle_tensor.storage().unwrap().storage_kind(),
        StorageKind::Structured
    );
    assert_eq!(middle_tensor.storage().unwrap().payload_dims(), &[2, 2]);
    assert_eq!(
        middle_tensor.storage().unwrap().axis_classes(),
        &[0, 0, 1, 1]
    );

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!(result_dense.distance(&state_dense).unwrap() < 1e-10);
}

#[test]
fn naive_apply_long_identity_chain_keeps_local_bonds_bounded() {
    // Keep this large enough that dense materialization would be infeasible,
    // while local exact apply remains linear in the chain length.
    let length = 24;
    let state_bond_dim = 2;
    let mpo_bond_dim = 1;
    let (state, operator, sites) =
        build_uniform_chain_state_and_identity_operator(length, state_bond_dim, mpo_bond_dim);

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert_eq!(result.node_count(), state.node_count());
    assert_eq!(result.edge_count(), state.edge_count());
    let mut edges: Vec<_> = result.site_index_network().edges().collect();
    edges.sort();
    for (left, right) in edges {
        let edge = result.edge_between(&left, &right).unwrap();
        assert!(
            result.bond_index(edge).unwrap().dim() <= state_bond_dim * mpo_bond_dim,
            "unexpected bond dimension on edge {left:?}-{right:?}"
        );
    }

    let site_indices: Vec<_> = sites.into_iter().map(|(_, idx)| idx).collect();
    let sample_points = [
        vec![0; length],
        vec![1; length],
        (0..length).map(|i| i % 2).collect::<Vec<_>>(),
    ];
    let mut values = Vec::with_capacity(length * sample_points.len());
    for point in &sample_points {
        values.extend(point.iter().copied());
    }
    let shape = [length, sample_points.len()];
    let value_ref = ColMajorArrayRef::new(&values, &shape).unwrap();
    let state_values = state.evaluate(&site_indices, value_ref).unwrap();
    let result_values = result.evaluate(&site_indices, value_ref).unwrap();

    for (state_value, result_value) in state_values.iter().zip(result_values.iter()) {
        assert!((state_value.real() - result_value.real()).abs() < 1e-10);
    }
}

#[test]
fn test_apply_linear_operator_partial() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 3-site state
    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        TensorDynLen::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 = TensorDynLen::from_dense(vec![b12.clone(), s2.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    let n2 = state.add_tensor("site2".to_string(), t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();

    // Create operator covering only site0 and site1 (partial)
    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test apply with partial operator (should extend with identity on site2)
    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), 3);
}

#[test]
fn test_apply_linear_operator_partial_preserves_site_index_set() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        TensorDynLen::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 = TensorDynLen::from_dense(vec![b12.clone(), s2.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    let n2 = state.add_tensor("site2".to_string(), t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();

    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    let (result_site_indices, _) = result.all_site_indices().unwrap();
    let result_site_ids: HashSet<_> = result_site_indices.iter().map(|idx| *idx.id()).collect();
    let expected_site_ids: HashSet<_> = [s0, s1, s2].iter().map(|idx| *idx.id()).collect();

    assert_eq!(result_site_ids, expected_site_ids);
    assert_eq!(result_site_indices.len(), expected_site_ids.len());
}

#[test]
fn test_apply_linear_operator_partial_shift_factorized_rooted_state_intermediate_indices() {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let spectator = make_index(2);
    let dense = TensorDynLen::from_dense(
        vec![s0.clone(), s1.clone(), spectator.clone()],
        vec![10.0, 30.0, 20.0, 40.0, 11.0, 31.0, 21.0, 41.0],
    )
    .unwrap();

    let mut nodes = HashMap::new();
    nodes.insert(0usize, vec![s0.clone()]);
    nodes.insert(1usize, vec![s1.clone()]);
    nodes.insert(2usize, vec![spectator.clone()]);
    let topology = crate::TreeTopology::new(nodes, vec![(0usize, 1usize), (1usize, 2usize)]);
    let state = crate::factorize_tensor_to_treetn(&dense, &topology, &2usize).unwrap();
    let mut state_edges = state.site_index_network().edges().collect::<Vec<_>>();
    state_edges.sort();
    assert_eq!(state_edges, vec![(0usize, 1usize), (1usize, 2usize)]);

    let mut mpo = TreeTN::<TensorDynLen, usize>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor(0usize, t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor(1usize, t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    input_mapping.insert(
        0usize,
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        1usize,
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    output_mapping.insert(
        0usize,
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        1usize,
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    let full_operator = extend_operator_to_full_space(&operator, &state).unwrap();
    let mut output_keys: Vec<_> = full_operator.output_mappings().keys().cloned().collect();
    output_keys.sort();
    assert_eq!(output_keys, vec![0usize, 1usize, 2usize]);
    let transformed_state = transform_state_to_input(&full_operator, &state).unwrap();
    let contracted_tensor = transformed_state
        .contract_naive(full_operator.mpo())
        .unwrap();

    let contracted_ids: HashSet<_> = contracted_tensor
        .external_indices()
        .iter()
        .map(|idx| *idx.id())
        .collect();
    let expected_ids: HashSet<_> = [
        *full_operator
            .get_output_mapping(&0)
            .unwrap()
            .internal_index
            .id(),
        *full_operator
            .get_output_mapping(&1)
            .unwrap()
            .internal_index
            .id(),
        *full_operator
            .get_output_mapping(&2)
            .unwrap()
            .internal_index
            .id(),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        contracted_ids,
        expected_ids,
        "input ids {:?}, output ids {:?}, contracted ids {:?}",
        [
            *full_operator
                .get_input_mapping(&0)
                .unwrap()
                .internal_index
                .id(),
            *full_operator
                .get_input_mapping(&1)
                .unwrap()
                .internal_index
                .id(),
        ],
        [
            *full_operator
                .get_output_mapping(&0)
                .unwrap()
                .internal_index
                .id(),
            *full_operator
                .get_output_mapping(&1)
                .unwrap()
                .internal_index
                .id(),
        ],
        contracted_tensor
            .external_indices()
            .iter()
            .map(|idx| *idx.id())
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        contracted_tensor.external_indices().len(),
        expected_ids.len()
    );

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let (result_site_indices, _) = result.all_site_indices().unwrap();
    let result_site_ids: HashSet<_> = result_site_indices.iter().map(|idx| *idx.id()).collect();
    let expected_true_ids: HashSet<_> = [s0, s1, spectator].iter().map(|idx| *idx.id()).collect();
    assert_eq!(result_site_ids, expected_true_ids);
    assert_eq!(result_site_indices.len(), expected_true_ids.len());
}

#[test]
fn test_apply_linear_operator_non_contiguous_chain() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[sites[0].clone(), sites[2].clone()]);

    assert_identity_application(&state, &operator);
}

#[test]
fn test_apply_linear_operator_non_contiguous_tree() {
    let (state, sites) = build_tree_state();
    let operator = build_identity_operator(&[sites[0].clone(), sites[4].clone()]);

    assert_identity_application(&state, &operator);
}

#[test]
fn test_apply_linear_operator_non_contiguous_chain_with_bonded_operator() {
    let (state, sites) = build_chain_state();
    let operator = build_bonded_identity_operator(&[sites[0].clone(), sites[2].clone()]);

    let result =
        crate::operator::apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), state.node_count());
}

#[test]
fn test_apply_linear_operator_non_contiguous_tree_with_bonded_operator() {
    let (state, sites) = build_tree_state();
    let operator = build_bonded_identity_operator(&[sites[0].clone(), sites[4].clone()]);

    let result =
        crate::operator::apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), state.node_count());
}

#[test]
fn test_apply_linear_operator_error_cases() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 2-site state
    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let b01 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();

    // Create operator with extra node not in state (should error)
    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s2_in = make_index(2); // site2 doesn't exist in state
    let s2_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t2_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s2_out.clone(), s2_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n2_mpo = mpo.add_tensor("site2".to_string(), t2_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n2_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "site2".to_string(),
        IndexMapping {
            true_index: s1.clone(), // Using s1 as true index for site2
            internal_index: s2_in.clone(),
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "site2".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s2_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Should error because operator has node "site2" not in state
    let result = apply_linear_operator(&operator, &state, ApplyOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_extend_operator_to_full_space_rejects_missing_state_node() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[("ghost".to_string(), sites[0].1.clone())]);

    let err = extend_operator_to_full_space(&operator, &state).unwrap_err();
    assert!(err.to_string().contains("missing from the state network"));
}

#[test]
fn test_extend_operator_to_full_space_uses_unchecked_branch_for_single_node_operator() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[sites[0].clone()]);

    let extended = extend_operator_to_full_space(&operator, &state).unwrap();
    assert_eq!(extended.mpo.node_count(), state.node_count());
    assert_eq!(extended.mpo.edge_count(), state.edge_count());
}

#[test]
fn test_compose_operator_along_state_paths_scalar_identity_gap() {
    let mut state_network: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let a = make_index(2);
    state_network
        .add_node(
            "A".to_string(),
            [a.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    state_network
        .add_node("B".to_string(), HashSet::new())
        .unwrap();
    state_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let operator = build_identity_operator(&[("A".to_string(), a.clone())]);
    let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
    gap_site_indices.insert("B".to_string(), Vec::new());

    let composed = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap();

    assert_eq!(composed.mpo.node_count(), 2);
    assert_eq!(composed.mpo.edge_count(), 1);
}

#[test]
fn test_compose_operator_along_state_paths_missing_gap_indices() {
    let (state, sites) = build_chain_state();
    let state_network = state.site_index_network().clone();
    let operator = build_bonded_identity_operator(&[sites[0].clone(), sites[2].clone()]);

    let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
    gap_site_indices.insert("site0".to_string(), Vec::new());
    gap_site_indices.insert("site2".to_string(), Vec::new());

    let err = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("missing gap indices"));
}

#[test]
fn test_compose_operator_along_state_paths_missing_state_node() {
    let mut state_network: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let a = make_index(2);
    state_network
        .add_node(
            "A".to_string(),
            [a.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();

    let ghost = make_index(2);
    let mut mpo = TreeTN::new();
    let a_in = make_index(2);
    let a_out = make_index(2);
    let g_in = make_index(2);
    let g_out = make_index(2);
    let bond = make_index(1);

    let t_a = TensorDynLen::from_dense(
        vec![a_out.clone(), a_in.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let t_g = TensorDynLen::from_dense(
        vec![bond.clone(), g_out.clone(), g_in.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let n_a = mpo.add_tensor("A".to_string(), t_a).unwrap();
    let n_g = mpo.add_tensor("Ghost".to_string(), t_g).unwrap();
    mpo.connect(n_a, &bond, n_g, &bond).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    input_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: a.clone(),
            internal_index: a_in.clone(),
        },
    );
    input_mapping.insert(
        "Ghost".to_string(),
        IndexMapping {
            true_index: ghost.clone(),
            internal_index: g_in.clone(),
        },
    );
    output_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: a.clone(),
            internal_index: a_out.clone(),
        },
    );
    output_mapping.insert(
        "Ghost".to_string(),
        IndexMapping {
            true_index: ghost.clone(),
            internal_index: g_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);
    let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

    let err = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("missing state node"));
}

#[test]
fn test_compose_operator_along_state_paths_no_path_between_operator_nodes() {
    let mut state_network: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let a = make_index(2);
    let c = make_index(2);
    state_network
        .add_node(
            "A".to_string(),
            [a.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    state_network
        .add_node(
            "C".to_string(),
            [c.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();

    let operator = build_bonded_identity_operator(&[
        ("A".to_string(), a.clone()),
        ("C".to_string(), c.clone()),
    ]);
    let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

    let err = compose_operator_along_state_paths(
        &operator,
        &state_network,
        &gap_site_indices,
        operator.input_mapping.clone(),
        operator.output_mapping.clone(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("no path between"));
}

#[test]
fn test_linear_operator_replace_indices() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    net.add_node(
        "N0".to_string(),
        [s0_in.clone(), s0_out.clone()]
            .into_iter()
            .collect::<HashSet<_>>(),
    )
    .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space).unwrap();

    let true_s0 = make_index(2);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    output_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_out.clone(),
        },
    );

    let lin_op = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test replace_indices
    let new_idx1 = make_index(2);
    let new_idx2 = make_index(2);
    let replaced = lin_op
        .replace_indices(
            std::slice::from_ref(&true_s0),
            std::slice::from_ref(&new_idx1),
        )
        .unwrap();
    assert_eq!(
        replaced
            .get_input_mapping(&"N0".to_string())
            .unwrap()
            .true_index,
        new_idx1
    );
    assert_eq!(
        replaced
            .get_output_mapping(&"N0".to_string())
            .unwrap()
            .true_index,
        new_idx1
    );

    // Replacing the same old index twice is ambiguous now that replaceind
    // updates all matching input/output mappings.
    let new_idx3 = make_index(2);
    let duplicate =
        lin_op.replace_indices(&[true_s0.clone(), true_s0.clone()], &[new_idx2, new_idx3]);
    assert!(duplicate.is_err());
    assert!(duplicate
        .unwrap_err()
        .to_string()
        .contains("Duplicate old index"));

    // Test replace_indices error case (length mismatch)
    let result = lin_op.replace_indices(
        std::slice::from_ref(&true_s0),
        &[new_idx1.clone(), new_idx1],
    );
    assert!(result.is_err());
}
