use super::*;

fn projector(pairs: impl IntoIterator<Item = (DynIndex, usize)>) -> Projector {
    Projector::from_pairs(pairs).unwrap()
}

fn subdomain(data: TensorTrain, projector: Projector) -> SubDomainTT {
    SubDomainTT::new(data, projector).unwrap()
}
use std::cmp::Ordering;
use tensor4all_core::index::Index;
use tensor4all_core::{IdxTensor, TagSet, TensorContractionLike};
use tensor4all_itensorlike::TensorTrainError;

fn make_index(size: usize) -> DynIndex {
    Index::new_dyn(size)
}

fn make_tensor(indices: Vec<DynIndex>) -> IdxTensor {
    let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
    let size: usize = dims.iter().product();
    let data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
    IdxTensor::from_dense(indices, data).unwrap()
}

/// Create shared indices for testing
fn make_shared_indices() -> (Vec<DynIndex>, DynIndex) {
    let s0 = make_index(2); // site 0
    let l01 = make_index(3); // link 0-1
    let s1 = make_index(2); // site 1
    (vec![s0, s1], l01)
}

/// Create a TT using the provided indices
fn make_tt_with_indices(site_inds: &[DynIndex], link_ind: &DynIndex) -> TensorTrain {
    let t0 = make_tensor(vec![site_inds[0].clone(), link_ind.clone()]);
    let t1 = make_tensor(vec![link_ind.clone(), site_inds[1].clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

fn make_simple_tt() -> (TensorTrain, Vec<DynIndex>) {
    let (site_inds, link_ind) = make_shared_indices();
    let tt = make_tt_with_indices(&site_inds, &link_ind);
    (tt, site_inds)
}

#[test]
fn test_partitioned_tt_creation() {
    // Create shared indices so both TTs have the same site indices
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 1)]));

    let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

    assert_eq!(partitioned.len(), 2);
    assert!(!partitioned.is_empty());
}

#[test]
fn deterministic_projector_comparator_is_equal_iff_projector_equal() {
    let base = make_index(2);
    let tagged = Index::new_with_tags(base.id, 2, TagSet::from_str("Site").unwrap());
    let primed = base.prime();
    let different_dimension = Index::new_with_tags(base.id, 5, base.tags.clone());

    let plain_projector = projector([(base.clone(), 0)]);
    let tagged_projector = projector([(tagged, 0)]);
    let primed_projector = projector([(primed, 0)]);
    let equal_projector = projector([(base, 0)]);
    let different_dimension_projector = projector([(different_dimension, 0)]);

    assert_ne!(plain_projector, tagged_projector);
    assert_ne!(plain_projector, primed_projector);
    assert_eq!(plain_projector, equal_projector);
    assert_eq!(plain_projector, different_dimension_projector);
    assert_ne!(
        plain_projector.canonical_cmp(&tagged_projector),
        Ordering::Equal
    );
    assert_ne!(
        plain_projector.canonical_cmp(&primed_projector),
        Ordering::Equal
    );
    assert_eq!(
        plain_projector.canonical_cmp(&equal_projector),
        Ordering::Equal
    );
    assert_eq!(
        plain_projector.canonical_cmp(&different_dimension_projector),
        Ordering::Equal
    );
}

#[test]
fn partitioned_tt_insert_overlap_is_transactional() {
    let (site_inds, link_ind) = make_shared_indices();
    let original = subdomain(
        make_tt_with_indices(&site_inds, &link_ind),
        projector([(site_inds[0].clone(), 0)]),
    );
    let overlapping = subdomain(
        make_tt_with_indices(&site_inds, &link_ind),
        projector([(site_inds[1].clone(), 0)]),
    );
    let original_projector = original.projector().clone();
    let mut partitioned = PartitionedTT::from_subdomain(original).unwrap();

    let error = partitioned.insert(overlapping).unwrap_err();

    assert!(matches!(error, PartitionedTTError::OverlappingProjectors));
    assert_eq!(partitioned.len(), 1);
    assert!(partitioned.contains(&original_projector));
}

#[test]
fn partitioned_tt_insert_exact_projector_replaces_key_and_value() {
    let (site_inds, link_ind) = make_shared_indices();
    let original_projector = projector([(site_inds[0].clone(), 1)]);
    let original = subdomain(
        make_tt_with_indices(&site_inds, &link_ind),
        original_projector.clone(),
    );
    let replacement_site = Index::new_with_tags(site_inds[0].id, 5, site_inds[0].tags.clone());
    let replacement_projector = projector([(replacement_site.clone(), 1)]);
    let replacement = subdomain(
        make_tt_with_indices(&[replacement_site.clone(), site_inds[1].clone()], &link_ind),
        replacement_projector.clone(),
    );
    let mut partitioned = PartitionedTT::from_subdomain(original).unwrap();

    let _ = partitioned.insert(replacement);

    let stored_projector = partitioned.projectors().next().unwrap();
    let stored_index = stored_projector.projected_indices().next().unwrap();
    assert_eq!(partitioned.len(), 1);
    assert_eq!(stored_index, &replacement_site);
    assert_eq!(
        partitioned.get(&replacement_projector).unwrap().projector(),
        &replacement_projector
    );
}

#[test]
fn test_partitioned_tt_empty() {
    let partitioned = PartitionedTT::new();

    assert_eq!(partitioned.len(), 0);
    assert!(partitioned.is_empty());
}

#[test]
fn test_partitioned_tt_overlapping_projectors() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[1].clone(), 0)]));

    let result = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]);
    assert!(matches!(
        result,
        Err(PartitionedTTError::OverlappingProjectors)
    ));
}

#[test]
fn test_partitioned_tt_norm() {
    let (tt, _) = make_simple_tt();
    let subdomain = SubDomainTT::from_tt(tt);
    let partitioned = PartitionedTT::from_subdomain(subdomain).unwrap();

    let norm = partitioned.norm().unwrap();
    assert!(norm > 0.0);
}

#[test]
fn test_partitioned_tt_append() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));
    let mut partitioned1 = PartitionedTT::from_subdomain(subdomain1).unwrap();

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 1)]));
    let partitioned2 = PartitionedTT::from_subdomain(subdomain2).unwrap();

    partitioned1.append(partitioned2).unwrap();

    assert_eq!(partitioned1.len(), 2);
}

#[test]
fn test_partitioned_tt_append_overlapping() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let projector1 = projector([(site_inds[0].clone(), 0)]);
    let subdomain1 = subdomain(tt1, projector1.clone());
    let mut partitioned1 = PartitionedTT::from_subdomain(subdomain1).unwrap();

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[1].clone(), 0)]));
    let partitioned2 = PartitionedTT::from_subdomain(subdomain2).unwrap();

    let result = partitioned1.append(partitioned2);
    assert!(matches!(
        result,
        Err(PartitionedTTError::OverlappingProjectors)
    ));
    assert_eq!(partitioned1.len(), 1);
    assert!(partitioned1.contains(&projector1));
}

#[test]
fn test_partitioned_tt_iteration() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 1)]));

    let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

    let count = partitioned.iter().count();
    assert_eq!(count, 2);

    let projector_count = partitioned.projectors().count();
    assert_eq!(projector_count, 2);
}

#[test]
fn test_partitioned_tt_index() {
    let (tt, site_inds) = make_simple_tt();
    let projector = projector([(site_inds[0].clone(), 0)]);
    let subdomain = subdomain(tt, projector.clone());
    let partitioned = PartitionedTT::from_subdomain(subdomain).unwrap();

    // Access by projector
    let retrieved = partitioned.get(&projector).unwrap();
    assert_eq!(retrieved.projector(), &projector);
}

/// Create contraction test indices:
/// TT1: s0 -- l01 -- s1
/// TT2: s1 -- l12 -- s2
fn make_contraction_indices() -> (DynIndex, DynIndex, DynIndex, DynIndex, DynIndex) {
    let s0 = make_index(2);
    let l01 = make_index(3);
    let s1 = make_index(2);
    let l12 = make_index(3);
    let s2 = make_index(2);
    (s0, l01, s1, l12, s2)
}

fn make_tt1(s0: &DynIndex, l01: &DynIndex, s1: &DynIndex) -> TensorTrain {
    let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

fn make_tt2(s1: &DynIndex, l12: &DynIndex, s2: &DynIndex) -> TensorTrain {
    let t0 = make_tensor(vec![s1.clone(), l12.clone()]);
    let t1 = make_tensor(vec![l12.clone(), s2.clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

fn project_dense_tensor_at_index(
    tensor: &IdxTensor,
    index: &DynIndex,
    projected_value: usize,
) -> IdxTensor {
    let indices = tensor.indices().to_vec();
    let axis = indices
        .iter()
        .position(|candidate| candidate == index)
        .unwrap();
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim).collect();
    let axis_stride = dims[..axis].iter().copied().product::<usize>().max(1);
    let axis_dim = dims[axis];
    let src_data = tensor.to_vec::<f64>().unwrap();
    let mut projected_data = vec![0.0_f64; src_data.len()];

    for (flat_idx, value) in src_data.iter().copied().enumerate() {
        let axis_value = (flat_idx / axis_stride) % axis_dim;
        if axis_value == projected_value {
            projected_data[flat_idx] = value;
        }
    }

    IdxTensor::from_dense(indices, projected_data).unwrap()
}

#[test]
fn test_partitioned_tt_contract_numerical() {
    let (s0, l01, s1, l12, s2) = make_contraction_indices();

    // Create PartitionedTT1 with 2 subdomains: s0=0 and s0=1
    let tt1_a = make_tt1(&s0, &l01, &s1);
    let tt1_b = make_tt1(&s0, &l01, &s1);
    let subdomain1_a = subdomain(tt1_a, projector([(s0.clone(), 0)]));
    let subdomain1_b = subdomain(tt1_b, projector([(s0.clone(), 1)]));
    let partitioned1 = PartitionedTT::from_subdomains(vec![subdomain1_a, subdomain1_b]).unwrap();

    // Create PartitionedTT2 with 2 subdomains: s2=0 and s2=1
    let tt2_a = make_tt2(&s1, &l12, &s2);
    let tt2_b = make_tt2(&s1, &l12, &s2);
    let subdomain2_a = subdomain(tt2_a, projector([(s2.clone(), 0)]));
    let subdomain2_b = subdomain(tt2_b, projector([(s2.clone(), 1)]));
    let partitioned2 = PartitionedTT::from_subdomains(vec![subdomain2_a, subdomain2_b]).unwrap();

    // Contract
    let options = ContractOptions::default();
    let result = partitioned1.contract(&partitioned2, &options).unwrap();

    // Result should have 4 subdomains: (s0=0,s2=0), (s0=0,s2=1), (s0=1,s2=0), (s0=1,s2=1)
    assert_eq!(result.len(), 4);

    // Verify each subdomain numerically
    for (proj, subdomain) in result.iter() {
        let contracted_full = subdomain.data().to_dense().unwrap();
        let contracted_data = contracted_full.to_vec::<f64>().unwrap();

        // Get the projected values
        let s0_val = proj.get(&s0).unwrap();
        let s2_val = proj.get(&s2).unwrap();

        // Compute expected by projecting full TTs
        let tt1 = make_tt1(&s0, &l01, &s1);
        let tt2 = make_tt2(&s1, &l12, &s2);
        let t1_full = tt1.to_dense().unwrap();
        let t2_full = tt2.to_dense().unwrap();

        // Project t1 to s0=s0_val
        let t1_proj = project_dense_tensor_at_index(&t1_full, &s0, s0_val);

        // Project t2 to s2=s2_val
        let t2_proj = project_dense_tensor_at_index(&t2_full, &s2, s2_val);

        let expected = t1_proj.contract_pair(&t2_proj).unwrap();
        let expected_data = expected.to_vec::<f64>().unwrap();

        assert_eq!(
            contracted_data.len(),
            expected_data.len(),
            "Size mismatch for projector {:?}",
            proj
        );
        for (i, (&actual, &exp)) in contracted_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-10,
                "Mismatch at index {} for projector {:?}: actual={}, expected={}",
                i,
                proj,
                actual,
                exp
            );
        }
    }
}

#[test]
fn test_subdomain_tt_norm_with_projector() {
    let (site_inds, link_ind) = make_shared_indices();

    // Create TT and get its full tensor
    let tt = make_tt_with_indices(&site_inds, &link_ind);
    let full_tensor = tt.to_dense().unwrap();
    let full_data = full_tensor.to_vec::<f64>().unwrap();

    // Create SubDomainTT with projector s0=0
    let projector = projector([(site_inds[0].clone(), 0)]);
    let subdomain = subdomain(tt.clone(), projector);

    // Current norm() returns the norm of the underlying TT data without projection
    // This test documents current behavior
    let norm_raw = subdomain.norm().unwrap();
    let tt_norm = tt.norm().unwrap();
    assert!((norm_raw - tt_norm).abs() < 1e-10);

    // Compute expected norm if projection were applied:
    // For s0=0, we keep only indices 0, 1 (first row of 2x2 matrix)
    let mut projected_sum_sq = 0.0;
    for &x in full_data.iter().take(site_inds[1].dim) {
        projected_sum_sq += x.powi(2);
    }
    let _expected_projected_norm = projected_sum_sq.sqrt();

    // Note: Current implementation returns raw TT norm, not projected norm
    // If we want projected norm, we'd need to modify the implementation
    // For now, this test just verifies current behavior is consistent
}

#[test]
fn test_partitioned_tt_add_preserves_tensor_train_error_variant() {
    let left_index = make_index(2);
    let right_index = make_index(3);
    let left = SubDomainTT::from_tt(TensorTrain::new(vec![make_tensor(vec![left_index])]).unwrap());
    let right =
        SubDomainTT::from_tt(TensorTrain::new(vec![make_tensor(vec![right_index])]).unwrap());
    let left_partitioned = PartitionedTT::from_subdomain(left).unwrap();
    let right_partitioned = PartitionedTT::from_subdomain(right).unwrap();

    let error = left_partitioned
        .add(&right_partitioned, &TruncateOptions::svd())
        .unwrap_err();
    match error {
        PartitionedTTError::TensorTrain { source } => {
            assert!(matches!(source, TensorTrainError::InvalidStructure { .. }));
        }
        other => panic!("expected preserved tensor-train error, got {other:?}"),
    }
}

#[test]
fn test_partitioned_tt_add_same_structure() {
    let (site_inds, link_ind) = make_shared_indices();

    // Create two PartitionedTTs with the same patch structure
    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));
    let partitioned1 = PartitionedTT::from_subdomain(subdomain1).unwrap();

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 0)]));
    let partitioned2 = PartitionedTT::from_subdomain(subdomain2).unwrap();

    // Add them
    let options = TruncateOptions::svd();
    let result = partitioned1.add(&partitioned2, &options).unwrap();

    // Result should have 1 subdomain (same projector)
    assert_eq!(result.len(), 1);

    // The sum should be 2x the original (same TT added to itself)
    let proj = projector([(site_inds[0].clone(), 0)]);
    let summed = result.get(&proj).unwrap();
    let summed_dense = summed.data().to_dense().unwrap();
    let summed_data = summed_dense.to_vec::<f64>().unwrap();

    let original = make_tt_with_indices(&site_inds, &link_ind);
    let original_dense = original.to_dense().unwrap();
    let original_data = original_dense.to_vec::<f64>().unwrap();

    for (i, (&s, &o)) in summed_data.iter().zip(original_data.iter()).enumerate() {
        assert!(
            (s - 2.0 * o).abs() < 1e-10,
            "Mismatch at index {}: summed={}, expected={}",
            i,
            s,
            2.0 * o
        );
    }
}

#[test]
fn test_partitioned_tt_add_missing_patch() {
    let (site_inds, link_ind) = make_shared_indices();

    // partitioned1 has patches for s0=0 and s0=1
    let tt1_a = make_tt_with_indices(&site_inds, &link_ind);
    let tt1_b = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1_a = subdomain(tt1_a, projector([(site_inds[0].clone(), 0)]));
    let subdomain1_b = subdomain(tt1_b, projector([(site_inds[0].clone(), 1)]));
    let partitioned1 = PartitionedTT::from_subdomains(vec![subdomain1_a, subdomain1_b]).unwrap();

    // partitioned2 has only patch for s0=0
    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 0)]));
    let partitioned2 = PartitionedTT::from_subdomain(subdomain2).unwrap();

    // Add them
    let options = TruncateOptions::svd();
    let result = partitioned1.add(&partitioned2, &options).unwrap();

    // Result should have 2 subdomains
    assert_eq!(result.len(), 2);

    // s0=0 patch should be summed (2x)
    let proj0 = projector([(site_inds[0].clone(), 0)]);
    let summed0 = result.get(&proj0).unwrap();
    let original = make_tt_with_indices(&site_inds, &link_ind);
    let original_norm = original.norm().unwrap();
    // Norm of 2*TT is 2*norm(TT)
    assert!((summed0.norm().unwrap() - 2.0 * original_norm).abs() < 1e-10);

    // s0=1 patch should be unchanged (only in partitioned1)
    let proj1 = projector([(site_inds[0].clone(), 1)]);
    let unchanged = result.get(&proj1).unwrap();
    assert!((unchanged.norm().unwrap() - original_norm).abs() < 1e-10);
}

#[test]
fn test_partitioned_tt_values() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 1)]));

    let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

    // Test values()
    let values_count = partitioned.values().count();
    assert_eq!(values_count, 2);

    // Check that each value has non-zero norm
    for subdomain in partitioned.values() {
        assert!(subdomain.norm().unwrap() > 0.0);
    }
}

#[test]
fn test_partitioned_tt_append_subdomains() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));
    let mut partitioned = PartitionedTT::from_subdomain(subdomain1).unwrap();

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 1)]));

    partitioned.append_subdomains(vec![subdomain2]).unwrap();
    assert_eq!(partitioned.len(), 2);
}

#[test]
fn test_partitioned_tt_append_subdomains_overlapping() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let projector1 = projector([(site_inds[0].clone(), 0)]);
    let subdomain1 = subdomain(tt1, projector1.clone());
    let mut partitioned = PartitionedTT::from_subdomain(subdomain1).unwrap();

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[1].clone(), 0)]));

    let result = partitioned.append_subdomains(vec![subdomain2]);
    assert!(matches!(
        result,
        Err(PartitionedTTError::OverlappingProjectors)
    ));
    assert_eq!(partitioned.len(), 1);
    assert!(partitioned.contains(&projector1));
}

#[test]
fn test_partitioned_tt_to_tensor_train_empty() {
    let partitioned = PartitionedTT::new();
    let result = partitioned.to_tensor_train();
    assert!(result.is_err());
}

#[test]
fn test_partitioned_tt_to_tensor_train_single() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt = make_tt_with_indices(&site_inds, &link_ind);
    let expected_dense = tt.to_dense().unwrap();
    let subdomain = SubDomainTT::from_tt(tt);
    let partitioned = PartitionedTT::from_subdomain(subdomain).unwrap();

    let combined = partitioned.to_tensor_train().unwrap();
    let combined_dense = combined.to_dense().unwrap();

    let diff = combined_dense.distance(&expected_dense).unwrap();
    assert!(diff < 1e-10);
}

#[test]
fn test_partitioned_tt_to_tensor_train_multiple() {
    let (site_inds, link_ind) = make_shared_indices();

    // Create two subdomains with different projectors
    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1.clone(), projector([(site_inds[0].clone(), 0)]));
    let subdomain2 = subdomain(tt2.clone(), projector([(site_inds[0].clone(), 1)]));
    let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

    let combined = partitioned.to_tensor_train().unwrap();

    // The combined TT should equal the sum of the two individual TTs
    let expected = tt1.add(&tt2).unwrap();
    let combined_dense = combined.to_dense().unwrap();
    let expected_dense = expected.to_dense().unwrap();

    let diff = combined_dense.distance(&expected_dense).unwrap();
    assert!(diff < 1e-10);
}

#[test]
fn test_partitioned_tt_into_iter() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[0].clone(), 1)]));

    let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

    // Test owned into_iter
    let collected: Vec<(Projector, SubDomainTT)> = partitioned.into_iter().collect();
    assert_eq!(collected.len(), 2);
}

#[test]
fn test_partitioned_tt_contract_with_duplicate_projectors() {
    // This test covers the branch in contract() where multiple subdomain pairs
    // produce the same output projector, triggering TT addition and truncation.
    //
    // Setup: Both PartitionedTTs partition over the *same* contracted index s1.
    // TT1 has patches for s1=0 and s1=1 (both have external index s0).
    // TT2 has patches for s1=0 and s1=1 (both have external index s2).
    // When contracting, (s1=0, s1=0) and (s1=1, s1=1) both produce a result
    // with an empty projector (since s1 is contracted away and neither s0 nor s2
    // is projected). This triggers the "existing projector" branch.
    let s0 = make_index(2);
    let l01 = make_index(3);
    let s1 = make_index(2);
    let l12 = make_index(3);
    let s2 = make_index(2);

    // TT1 partitioned by s1: patches for s1=0 and s1=1
    let tt1_a = make_tt1(&s0, &l01, &s1);
    let tt1_b = make_tt1(&s0, &l01, &s1);
    let subdomain1_a = subdomain(tt1_a, projector([(s1.clone(), 0)]));
    let subdomain1_b = subdomain(tt1_b, projector([(s1.clone(), 1)]));
    let partitioned1 = PartitionedTT::from_subdomains(vec![subdomain1_a, subdomain1_b]).unwrap();

    // TT2 partitioned by s1: patches for s1=0 and s1=1
    let tt2_a = make_tt2(&s1, &l12, &s2);
    let tt2_b = make_tt2(&s1, &l12, &s2);
    let subdomain2_a = subdomain(tt2_a, projector([(s1.clone(), 0)]));
    let subdomain2_b = subdomain(tt2_b, projector([(s1.clone(), 1)]));
    let partitioned2 = PartitionedTT::from_subdomains(vec![subdomain2_a, subdomain2_b]).unwrap();

    // Contract: (s1=0 x s1=0) and (s1=1 x s1=1) both have compatible projectors
    // and produce results with the same output projector (empty, since s1 is contracted).
    // The second result triggers TT addition with the first.
    let options = ContractOptions::default();
    let result = partitioned1.contract(&partitioned2, &options).unwrap();

    // There should be at least one subdomain (the duplicate projectors get merged)
    assert!(!result.is_empty());

    // Verify the result is non-zero
    assert!(result.norm().unwrap() > 0.0);
}

#[test]
fn test_partitioned_tt_contract_with_rtol_and_max_bond_dim() {
    // Exercise the contract() branches that pass rtol and max_bond_dim to TruncateOptions
    let s0 = make_index(2);
    let l01 = make_index(3);
    let s1 = make_index(2);
    let l12 = make_index(3);
    let s2 = make_index(2);

    // Same setup as above to trigger the duplicate-projector branch
    let tt1_a = make_tt1(&s0, &l01, &s1);
    let tt1_b = make_tt1(&s0, &l01, &s1);
    let subdomain1_a = subdomain(tt1_a, projector([(s1.clone(), 0)]));
    let subdomain1_b = subdomain(tt1_b, projector([(s1.clone(), 1)]));
    let partitioned1 = PartitionedTT::from_subdomains(vec![subdomain1_a, subdomain1_b]).unwrap();

    let tt2_a = make_tt2(&s1, &l12, &s2);
    let tt2_b = make_tt2(&s1, &l12, &s2);
    let subdomain2_a = subdomain(tt2_a, projector([(s1.clone(), 0)]));
    let subdomain2_b = subdomain(tt2_b, projector([(s1.clone(), 1)]));
    let partitioned2 = PartitionedTT::from_subdomains(vec![subdomain2_a, subdomain2_b]).unwrap();

    let options = ContractOptions::default()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-12))
        .with_max_bond_dim(10);
    let result = partitioned1.contract(&partitioned2, &options).unwrap();

    assert!(!result.is_empty());
    assert!(result.norm().unwrap() > 0.0);
}

#[test]
fn test_partitioned_tt_add_disjoint_patches() {
    // Test add() where partitioned1 has only s0=0 and partitioned2 has only s0=1
    // This exercises the "only in other" branch (line 240) and the "only in self" branch
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1.clone(), projector([(site_inds[0].clone(), 0)]));
    let partitioned1 = PartitionedTT::from_subdomain(subdomain1).unwrap();

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2.clone(), projector([(site_inds[0].clone(), 1)]));
    let partitioned2 = PartitionedTT::from_subdomain(subdomain2).unwrap();

    let options = TruncateOptions::svd();
    let result = partitioned1.add(&partitioned2, &options).unwrap();

    // Should have 2 subdomains (no overlap, both copied)
    assert_eq!(result.len(), 2);

    // Each should have the same norm as the original
    let proj0 = projector([(site_inds[0].clone(), 0)]);
    let proj1 = projector([(site_inds[0].clone(), 1)]);
    assert!((result.get(&proj0).unwrap().norm().unwrap() - tt1.norm().unwrap()).abs() < 1e-10);
    assert!((result.get(&proj1).unwrap().norm().unwrap() - tt2.norm().unwrap()).abs() < 1e-10);
}

#[test]
fn test_partitioned_tt_add_overlapping_fails() {
    let (site_inds, link_ind) = make_shared_indices();

    // partitioned1 has patch for s0=0
    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = subdomain(tt1, projector([(site_inds[0].clone(), 0)]));
    let partitioned1 = PartitionedTT::from_subdomain(subdomain1).unwrap();

    // partitioned2 has patch for s1=0 (overlaps with s0=0 since they're compatible)
    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = subdomain(tt2, projector([(site_inds[1].clone(), 0)]));
    let partitioned2 = PartitionedTT::from_subdomain(subdomain2).unwrap();

    // Add should fail because projectors are compatible (not disjoint)
    let options = TruncateOptions::svd();
    let result = partitioned1.add(&partitioned2, &options);
    assert!(result.is_err());
}
