use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::{
    contraction::{ContractionMethod, ContractionOptions},
    TreeTN, TruncationOptions,
};

fn subdomain(site: &DynIndex, values: [f64; 2], coordinate: usize) -> SubDomainTreeTN {
    let tensor = IdxTensor::from_dense(vec![site.clone()], values.to_vec()).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    SubDomainTreeTN::new(
        tree,
        Projector::from_pairs([(site.clone(), coordinate)]).unwrap(),
    )
    .unwrap()
}

fn unprojected(site: &DynIndex, values: Vec<f64>) -> SubDomainTreeTN {
    let tensor = IdxTensor::from_dense(vec![site.clone()], values).unwrap();
    SubDomainTreeTN::from_treetn(TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap()).unwrap()
}

fn complex_subdomain(site: &DynIndex, coordinate: usize) -> SubDomainTreeTN {
    let tensor = IdxTensor::from_dense(
        vec![site.clone()],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0e12, -2.0e12)],
    )
    .unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    SubDomainTreeTN::new(
        tree,
        Projector::from_pairs([(site.clone(), coordinate)]).unwrap(),
    )
    .unwrap()
}

#[test]
fn partition_replaces_exact_keys_and_rejects_overlaps_transactionally() {
    let site = DynIndex::new_dyn(2);
    let mut partition = PartitionedTreeTN::from_subdomains(vec![
        subdomain(&site, [1.0, 2.0], 0),
        subdomain(&site, [3.0, 4.0], 1),
    ])
    .unwrap();

    let replacement = subdomain(&site, [10.0, 20.0], 0);
    partition.insert(replacement).unwrap();
    assert_eq!(partition.len(), 2);
    let stored = partition
        .get(&Projector::from_pairs([(site.clone(), 0)]).unwrap())
        .unwrap();
    assert_eq!(
        stored
            .data()
            .tensor(stored.data().node_index(&0).unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![10.0, 0.0]
    );

    let before = partition.clone();
    let overlapping = unprojected(&site, vec![7.0, 8.0]);
    assert!(partition.insert(overlapping).is_err());
    assert_eq!(partition.len(), before.len());
    assert!(partition.contains(&Projector::from_pairs([(site.clone(), 0)]).unwrap()));
    assert!(partition.contains(&Projector::from_pairs([(site, 1)]).unwrap()));
}

#[test]
fn partition_norm_sum_and_deterministic_treetn_sum_use_eager_values() {
    let site = DynIndex::new_dyn(2);
    let partition = PartitionedTreeTN::from_subdomains(vec![
        subdomain(&site, [3.0, 1.0e12], 0),
        subdomain(&site, [1.0e12, 4.0], 1),
    ])
    .unwrap();

    assert!((partition.norm_squared().unwrap() - 25.0).abs() < 1.0e-12);
    assert!((partition.norm().unwrap() - 5.0).abs() < 1.0e-12);

    let first = partition.to_treetn().unwrap();
    let second = partition.to_treetn().unwrap();
    let first_tensor = first.tensor(first.node_index(&0).unwrap()).unwrap();
    let second_tensor = second.tensor(second.node_index(&0).unwrap()).unwrap();
    assert_eq!(first_tensor.to_vec::<f64>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(second_tensor.to_vec::<f64>().unwrap(), vec![3.0, 4.0]);

    let empty = PartitionedTreeTN::<usize>::new();
    assert!(matches!(
        empty.to_treetn(),
        Err(PartitionedTreeTNError::Empty)
    ));
}

#[test]
fn partition_supports_complex_dtype_and_rejects_mixed_dtype_before_mutation() {
    let site = DynIndex::new_dyn(2);
    let complex = PartitionedTreeTN::from_subdomain(complex_subdomain(&site, 0)).unwrap();
    assert!((complex.norm_squared().unwrap() - 25.0).abs() < 1.0e-12);

    let real = PartitionedTreeTN::from_subdomain(subdomain(&site, [2.0, 3.0], 1)).unwrap();
    assert!(matches!(
        PartitionedTreeTN::from_subdomains(vec![
            complex.values().next().unwrap().clone(),
            real.values().next().unwrap().clone(),
        ]),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));

    let before = complex.clone();
    let mut target = complex.clone();
    assert!(matches!(
        target.append(real.clone()),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
    assert_eq!(target.len(), before.len());

    assert!(matches!(
        complex.add(&real, &0, TruncationOptions::default()),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        complex.contract(&real, &0, ContractionOptions::default()),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
}

#[test]
fn partition_addition_allows_missing_keys_and_rejects_overlapping_layouts() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [2.0, 100.0], 0)).unwrap();
    let right = PartitionedTreeTN::from_subdomain(subdomain(&site, [300.0, 4.0], 1)).unwrap();

    let sum = left.add(&right, &0, TruncationOptions::default()).unwrap();
    assert_eq!(sum.len(), 2);
    let dense = sum.to_treetn().unwrap();
    assert_eq!(
        dense
            .tensor(dense.node_index(&0).unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![2.0, 4.0]
    );

    let overlap = PartitionedTreeTN::from_subdomain(unprojected(&site, vec![1.0, 2.0])).unwrap();
    let before = left.clone();
    assert!(matches!(
        left.add(&overlap, &0, TruncationOptions::default()),
        Err(PartitionedTreeTNError::OverlappingProjectors)
    ));
    assert_eq!(left.len(), before.len());
}

#[test]
fn partition_addition_validates_center_and_options_before_shortcuts() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let right = PartitionedTreeTN::new();

    assert!(matches!(
        left.add(&right, &0, TruncationOptions::default()),
        Err(PartitionedTreeTNError::Empty)
    ));
    assert!(matches!(
        left.add(
            &left,
            &99,
            TruncationOptions::default().with_max_bond_dim(0),
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        left.add(&left, &99, TruncationOptions::default()),
        Err(PartitionedTreeTNError::InvalidCenter)
    ));
}

#[test]
fn partition_append_rejects_cross_overlap_transactionally() {
    let site = DynIndex::new_dyn(2);
    let mut target = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let before = target.clone();
    let other = PartitionedTreeTN::from_subdomain(unprojected(&site, vec![9.0, 10.0])).unwrap();

    assert!(matches!(
        target.append(other),
        Err(PartitionedTreeTNError::OverlappingProjectors)
    ));
    assert_eq!(target.len(), before.len());
    assert!(target.contains(&Projector::from_pairs([(site, 0)]).unwrap()));
}

fn shared_contract_patch(
    shared: &DynIndex,
    external: Option<&DynIndex>,
    shared_value: usize,
    external_value: Option<usize>,
    values: Vec<f64>,
) -> SubDomainTreeTN {
    let mut indices = vec![shared.clone()];
    let mut pairs = vec![(shared.clone(), shared_value)];
    if let Some(external) = external {
        indices.push(external.clone());
        pairs.push((external.clone(), external_value.unwrap_or(0)));
    }
    let tensor = IdxTensor::from_dense(indices, values).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    SubDomainTreeTN::new(tree, Projector::from_pairs(pairs).unwrap()).unwrap()
}

#[test]
fn contraction_prunes_contracted_projectors_and_retains_external_projectors() {
    let shared = DynIndex::new_dyn(2);
    let left_external = DynIndex::new_dyn(2);
    let right_external = DynIndex::new_dyn(2);
    let left = shared_contract_patch(
        &shared,
        Some(&left_external),
        0,
        Some(1),
        vec![0.0, 0.0, 0.0, 7.0],
    );
    let right = shared_contract_patch(
        &shared,
        Some(&right_external),
        0,
        Some(0),
        vec![5.0, 0.0, 0.0, 0.0],
    );

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap()
        .unwrap();
    assert_eq!(result.projector().get(&shared), None);
    assert_eq!(result.projector().get(&left_external), Some(1));
    assert_eq!(result.projector().get(&right_external), Some(0));
    assert!(result.all_indices().contains(&left_external));
    assert!(result.all_indices().contains(&right_external));
}

#[test]
fn partition_contraction_combines_duplicate_output_projectors_strictly() {
    let shared = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomains(vec![
        shared_contract_patch(&shared, None, 0, None, vec![2.0, 100.0]),
        shared_contract_patch(&shared, None, 1, None, vec![200.0, 3.0]),
    ])
    .unwrap();
    let right = PartitionedTreeTN::from_subdomains(vec![
        shared_contract_patch(&shared, None, 0, None, vec![5.0, 600.0]),
        shared_contract_patch(&shared, None, 1, None, vec![700.0, 7.0]),
    ])
    .unwrap();

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    assert_eq!(result.len(), 1);
    let dense = result.to_treetn().unwrap();
    let tensor = dense.tensor(dense.node_index(&0).unwrap()).unwrap();
    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![31.0]);
}

#[test]
fn partition_contraction_rejects_topology_site_assignment_dtype_and_dense_options() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let right_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0]).unwrap()],
        vec![1usize],
    )
    .unwrap();
    let right = PartitionedTreeTN::from_subdomain(
        SubDomainTreeTN::new(
            right_tree,
            Projector::from_pairs([(site.clone(), 0)]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        left.contract(&right, &1, ContractionOptions::default()),
        Err(PartitionedTreeTNError::TopologyMismatch)
    ));

    assert!(matches!(
        left.contract(&left, &0, ContractionOptions::new(ContractionMethod::Naive),),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
}

#[test]
fn partition_contract_duplicate_failure_does_not_mutate_inputs() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let right = PartitionedTreeTN::from_subdomain(subdomain(&site, [3.0, 4.0], 1)).unwrap();
    let left_before = left.clone();
    let right_before = right.clone();
    let _ = left.contract(&right, &0, ContractionOptions::default());
    assert_eq!(left.len(), left_before.len());
    assert_eq!(right.len(), right_before.len());
}
