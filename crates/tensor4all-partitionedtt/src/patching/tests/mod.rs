use super::*;
use crate::projector::Projector;
use approx::assert_abs_diff_eq;
use tensor4all_core::index::Index;
use tensor4all_core::TensorDynLen;
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

fn make_index(size: usize) -> DynIndex {
    Index::new_dyn(size)
}

fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
    let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
    let size: usize = dims.iter().product();
    let data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

/// Create shared indices for testing
fn make_shared_indices() -> (Vec<DynIndex>, DynIndex) {
    let s0 = make_index(2);
    let l01 = make_index(3);
    let s1 = make_index(2);
    (vec![s0, s1], l01)
}

/// Create a TT using the provided indices
fn make_tt_with_indices(site_inds: &[DynIndex], link_ind: &DynIndex) -> TensorTrain {
    let t0 = make_tensor(vec![site_inds[0].clone(), link_ind.clone()]);
    let t1 = make_tensor(vec![link_ind.clone(), site_inds[1].clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

fn make_scaled_rank_one_tt(site_inds: &[DynIndex], scale: f64) -> TensorTrain {
    let link = make_index(1);
    let t0 = TensorDynLen::from_dense(
        vec![site_inds[0].clone(), link.clone()],
        vec![scale; site_inds[0].dim],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![link, site_inds[1].clone()],
        vec![1.0; site_inds[1].dim],
    )
    .unwrap();
    TensorTrain::new(vec![t0, t1]).unwrap()
}

fn make_selector_middle_site_tt() -> (TensorTrain, Vec<DynIndex>) {
    let s0 = make_index(4);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let l01 = make_index(2);
    let l12 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), l01.clone()], vec![1.0; 8]).unwrap();
    let mut center_data = vec![0.0; 8];
    for channel in 0..2 {
        let flat = channel + 2 * channel + 4 * channel;
        center_data[flat] = 1.0;
    }
    let t1 = TensorDynLen::from_dense(vec![l01, s1.clone(), l12.clone()], center_data).unwrap();
    let t2 = TensorDynLen::from_dense(vec![l12, s2.clone()], vec![1.0; 4]).unwrap();

    (
        TensorTrain::new(vec![t0, t1, t2]).unwrap(),
        vec![s0, s1, s2],
    )
}

fn make_overcomplete_rank_one_tt(site_inds: &[DynIndex], link_ind: &DynIndex) -> TensorTrain {
    let mut left = vec![0.0; site_inds[0].dim * link_ind.dim];
    for (x, value) in left.iter_mut().take(site_inds[0].dim).enumerate() {
        *value = (x + 1) as f64;
    }
    let mut right = vec![0.0; link_ind.dim * site_inds[1].dim];
    for (y, chunk) in right.chunks_exact_mut(link_ind.dim).enumerate() {
        chunk[0] = (y + 1) as f64;
    }
    let t0 = TensorDynLen::from_dense(vec![site_inds[0].clone(), link_ind.clone()], left).unwrap();
    let t1 = TensorDynLen::from_dense(vec![link_ind.clone(), site_inds[1].clone()], right).unwrap();
    TensorTrain::new(vec![t0, t1]).unwrap()
}

fn make_contract_tt(
    s0: &DynIndex,
    l01: &DynIndex,
    s1: &DynIndex,
    l12: &DynIndex,
    s2: &DynIndex,
) -> TensorTrain {
    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
    let t2 = make_tensor(vec![l12.clone(), s2.clone()]);
    TensorTrain::new(vec![t0, t1, t2]).unwrap()
}

#[test]
fn test_add_with_patching_simple() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 1)]));

    let options = PatchingOptions::default();
    let result = add_with_patching(vec![subdomain1, subdomain2], &options).unwrap();

    assert_eq!(result.len(), 2);
}

#[test]
fn test_add_with_patching_splits_unprojected_patch_when_bond_hits_cap() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = SubDomainTT::from_tt(tt1);

    let options = PatchingOptions {
        rtol: 1e-12,
        max_bond_dim: 1,
        patch_order: vec![site_inds[0].clone()],
        split_strategy: PatchSplitStrategy::Sequential,
    };

    let result = add_with_patching(vec![subdomain1], &options).unwrap();
    let proj0 = Projector::from_pairs([(site_inds[0].clone(), 0)]);
    let proj1 = Projector::from_pairs([(site_inds[0].clone(), 1)]);

    assert_eq!(result.len(), 2);
    assert!(result.contains(&proj0));
    assert!(result.contains(&proj1));
    for subdomain in result.values() {
        assert!(subdomain.max_bond_dim() <= options.max_bond_dim);
        assert!(subdomain.budget_squared().is_some());
    }
}

#[test]
fn test_add_with_patching_truncates_before_deciding_to_split() {
    let (site_inds, link_ind) = make_shared_indices();
    let tt = make_overcomplete_rank_one_tt(&site_inds, &link_ind);
    assert_eq!(tt.max_bond_dim(), 3);

    let options = PatchingOptions {
        rtol: 0.0,
        max_bond_dim: 2,
        patch_order: vec![site_inds[0].clone()],
        split_strategy: PatchSplitStrategy::Sequential,
    };

    let result = add_with_patching(vec![SubDomainTT::from_tt(tt)], &options).unwrap();

    assert_eq!(result.len(), 1);
    let only_patch = result.values().next().unwrap();
    assert!(only_patch.projector().is_empty());
    assert_eq!(only_patch.max_bond_dim(), 2);
}

#[test]
fn test_truncate_adaptive_drops_patch_below_volume_budget() {
    let site_inds = vec![make_index(2), make_index(2)];
    let high_proj = Projector::from_pairs([(site_inds[0].clone(), 0)]);
    let low_proj = Projector::from_pairs([(site_inds[0].clone(), 1)]);

    let high = SubDomainTT::new(make_scaled_rank_one_tt(&site_inds, 10.0), high_proj.clone());
    let low = SubDomainTT::new(make_scaled_rank_one_tt(&site_inds, 0.01), low_proj.clone());
    let partitioned = PartitionedTT::from_subdomains(vec![high, low]).unwrap();

    let result = truncate_adaptive(&partitioned, 0.01, 4).unwrap();

    assert_eq!(result.len(), 1);
    assert!(result.contains(&high_proj));
    assert!(!result.contains(&low_proj));
}

#[test]
fn test_truncate_adaptive_assigns_volume_proportional_budgets() {
    let site_inds = vec![make_index(2), make_index(2)];
    let wide_proj = Projector::from_pairs([(site_inds[0].clone(), 0)]);
    let narrow_proj = Projector::from_pairs([(site_inds[0].clone(), 1), (site_inds[1].clone(), 0)]);

    let wide = SubDomainTT::new(make_scaled_rank_one_tt(&site_inds, 10.0), wide_proj.clone());
    let narrow = SubDomainTT::new(
        make_scaled_rank_one_tt(&site_inds, 10.0),
        narrow_proj.clone(),
    );
    let partitioned = PartitionedTT::from_subdomains(vec![wide, narrow]).unwrap();

    let result = truncate_adaptive(&partitioned, 1e-12, 4).unwrap();
    let wide_budget = result
        .get(&wide_proj)
        .and_then(SubDomainTT::budget_squared)
        .unwrap();
    let narrow_budget = result
        .get(&narrow_proj)
        .and_then(SubDomainTT::budget_squared)
        .unwrap();

    assert_abs_diff_eq!(wide_budget / narrow_budget, 2.0, epsilon = 1e-12);
}

#[test]
fn test_truncate_adaptive_rejects_invalid_options() {
    let empty = PartitionedTT::new();

    let bad_rtol = truncate_adaptive(&empty, f64::NAN, 1).unwrap_err();
    assert!(matches!(bad_rtol, PartitionedTTError::InvalidOptions(_)));

    let bad_rank = truncate_adaptive(&empty, 1e-12, 0).unwrap_err();
    assert!(matches!(bad_rank, PartitionedTTError::InvalidOptions(_)));
}

#[test]
fn test_contract_adaptive_retruncates_output_with_corrected_norm() {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let l01 = make_index(3);
    let l12 = make_index(3);
    let r01 = make_index(3);
    let r12 = make_index(3);
    let left = PartitionedTT::from_subdomain(SubDomainTT::from_tt(make_contract_tt(
        &s0, &l01, &s1, &l12, &s2,
    )));
    let right = PartitionedTT::from_subdomain(SubDomainTT::from_tt(make_contract_tt(
        &s0, &r01, &s1, &r12, &s2,
    )));
    let patching = PatchingOptions {
        rtol: 0.0,
        max_bond_dim: 1,
        patch_order: vec![s0],
        split_strategy: PatchSplitStrategy::Sequential,
    };

    let result = contract_adaptive(&left, &right, &ContractOptions::default(), &patching).unwrap();

    assert_eq!(result.len(), 1);
    assert!(result
        .values()
        .all(|subdomain| subdomain.max_bond_dim() <= patching.max_bond_dim));
}

#[test]
fn test_exact_parameter_gain_strategy_can_override_sequential_order() {
    let (tt, site_inds) = make_selector_middle_site_tt();
    let subdomain = SubDomainTT::from_tt(tt).with_budget_squared(1e-20);

    let sequential = PatchingOptions {
        rtol: 1e-12,
        max_bond_dim: 1,
        patch_order: vec![site_inds[0].clone(), site_inds[1].clone()],
        split_strategy: PatchSplitStrategy::Sequential,
    };
    let exact_gain = PatchingOptions {
        split_strategy: PatchSplitStrategy::ExactParameterGain,
        ..sequential.clone()
    };

    assert_eq!(
        choose_split_index(&subdomain, &sequential).unwrap(),
        Some(site_inds[0].clone())
    );
    assert_eq!(
        choose_split_index(&subdomain, &exact_gain).unwrap(),
        Some(site_inds[1].clone())
    );
}
