use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use num_complex::Complex64;
use tensor4all_core::index::{Index, TagSet};
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{PartitionedTreeTNError, Projector, SubDomainTreeTN};
use tensor4all_treetn::{contraction::ContractionOptions, TreeTN, TruncationOptions};

fn projector(pairs: impl IntoIterator<Item = (DynIndex, usize)>) -> Projector {
    Projector::from_pairs(pairs).unwrap()
}

fn one_node_f64() -> (TreeTN<IdxTensor, String>, DynIndex, DynIndex) {
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![left.clone(), right.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    (
        TreeTN::from_tensors(vec![tensor], vec!["root".to_string()]).unwrap(),
        left,
        right,
    )
}

fn chain_f64() -> (TreeTN<IdxTensor, String>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = IdxTensor::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(vec![b12, s2.clone()], vec![1.0; 4]).unwrap();
    (
        TreeTN::from_tensors(
            vec![t0, t1, t2],
            vec![
                "left".to_string(),
                "middle".to_string(),
                "right".to_string(),
            ],
        )
        .unwrap(),
        vec![s0, s1, s2],
    )
}

fn branched_f64() -> (TreeTN<IdxTensor, String>, Vec<DynIndex>) {
    let center_site = DynIndex::new_dyn(2);
    let leaf_sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let center = IdxTensor::from_dense(
        vec![
            center_site.clone(),
            bonds[0].clone(),
            bonds[1].clone(),
            bonds[2].clone(),
        ],
        vec![1.0; 16],
    )
    .unwrap();
    let leaves: Vec<_> = leaf_sites
        .iter()
        .zip(&bonds)
        .map(|(site, bond)| {
            IdxTensor::from_dense(vec![bond.clone(), site.clone()], vec![1.0; 4]).unwrap()
        })
        .collect();
    let mut tensors = vec![center];
    tensors.extend(leaves);
    (
        TreeTN::from_tensors(
            tensors,
            vec![
                "center".to_string(),
                "leaf0".to_string(),
                "leaf1".to_string(),
                "leaf2".to_string(),
            ],
        )
        .unwrap(),
        std::iter::once(center_site).chain(leaf_sites).collect(),
    )
}

#[test]
fn constructs_one_node_chain_and_branched_subdomains() {
    let (one_node, _, _) = one_node_f64();
    let one = SubDomainTreeTN::from_treetn(one_node).unwrap();
    assert_eq!(one.node_count(), 1);
    assert_eq!(one.site_index_network().edge_count(), 0);

    let (chain, sites) = chain_f64();
    let chain = SubDomainTreeTN::new(chain, projector([(sites[0].clone(), 1)])).unwrap();
    assert_eq!(chain.node_count(), 3);
    assert_eq!(chain.site_index_network().edge_count(), 2);
    assert_eq!(chain.all_indices().len(), 3);

    let (branched, sites) = branched_f64();
    let branched = SubDomainTreeTN::new(branched, projector([(sites[3].clone(), 0)])).unwrap();
    assert_eq!(branched.node_count(), 4);
    assert_eq!(branched.site_index_network().edge_count(), 3);
    assert_eq!(branched.all_indices().len(), 4);
}

#[test]
fn supports_multiple_site_indices_on_one_node_and_retains_axes() {
    let (tree, site0, site1) = one_node_f64();
    let original = tree
        .tensor(tree.node_index(&"root".to_string()).unwrap())
        .unwrap()
        .clone();
    let subdomain = SubDomainTreeTN::new(tree, projector([(site0.clone(), 1)])).unwrap();
    let masked = subdomain
        .data()
        .tensor(subdomain.data().node_index(&"root".to_string()).unwrap())
        .unwrap();

    assert_eq!(masked.indices(), original.indices());
    assert_eq!(masked.to_vec::<f64>().unwrap(), vec![0.0, 2.0, 0.0, 4.0]);
    assert!(subdomain.all_indices().contains(&site0));
    assert!(subdomain.all_indices().contains(&site1));
}

#[test]
fn eager_mask_handles_f64_c64_and_large_outside_values() {
    let site = DynIndex::new_dyn(3);
    let f64_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0, 1.0e12, -2.0e12]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let f64_subdomain = SubDomainTreeTN::new(f64_tree, projector([(site.clone(), 0)])).unwrap();
    let f64_tensor = f64_subdomain
        .data()
        .tensor(f64_subdomain.data().node_index(&0).unwrap())
        .unwrap();
    assert_eq!(f64_tensor.to_vec::<f64>().unwrap(), vec![1.0, 0.0, 0.0]);
    assert!((f64_subdomain.norm().unwrap() - 1.0).abs() < 1.0e-12);
    assert!((f64_subdomain.norm_squared().unwrap() - 1.0).abs() < 1.0e-12);

    let complex_site = DynIndex::new_dyn(2);
    let complex_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![complex_site.clone()],
            vec![Complex64::new(3.0, 4.0), Complex64::new(1.0e9, -2.0e9)],
        )
        .unwrap()],
        vec![0usize],
    )
    .unwrap();
    let complex_subdomain =
        SubDomainTreeTN::new(complex_tree, projector([(complex_site.clone(), 0)])).unwrap();
    let complex_tensor = complex_subdomain
        .data()
        .tensor(complex_subdomain.data().node_index(&0).unwrap())
        .unwrap();
    assert_eq!(
        complex_tensor.to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)]
    );
    assert!((complex_subdomain.norm_squared().unwrap() - 25.0).abs() < 1.0e-12);
}

#[test]
fn project_only_adds_compatible_restrictions_without_mutating_source() {
    let (tree, site0, site1) = one_node_f64();
    let source = SubDomainTreeTN::new(tree, projector([(site0.clone(), 0)])).unwrap();
    let source_values = source
        .data()
        .tensor(source.data().node_index(&"root".to_string()).unwrap())
        .unwrap()
        .to_vec::<f64>()
        .unwrap();

    let projected = source
        .project(&projector([(site1.clone(), 1)]))
        .unwrap()
        .unwrap();
    assert_eq!(projected.projector().get(&site0), Some(0));
    assert_eq!(projected.projector().get(&site1), Some(1));
    assert_eq!(source.projector().get(&site1), None);
    assert_eq!(
        source
            .data()
            .tensor(source.data().node_index(&"root".to_string()).unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        source_values
    );

    assert!(source
        .project(&projector([(site0.clone(), 1)]))
        .unwrap()
        .is_none());
    assert_eq!(source.projector().get(&site0), Some(0));
}

#[test]
fn norms_clone_before_canonicalization_and_rebuild_clears_metadata() {
    let (tree, site0, _) = one_node_f64();
    let mut canonical = tree.clone();
    canonical
        .set_canonical_region(["root".to_string()])
        .unwrap();
    assert!(!canonical.canonical_region().is_empty());

    let subdomain = SubDomainTreeTN::new(canonical, projector([(site0, 1)])).unwrap();
    assert!(subdomain.data().canonical_region().is_empty());
    let _ = subdomain.norm().unwrap();
    let _ = subdomain.norm_squared().unwrap();
    assert!(subdomain.data().canonical_region().is_empty());
}

#[test]
fn preserves_ad_and_structured_storage_through_masking() {
    let site = DynIndex::new_dyn(2);
    let source = IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0])
        .unwrap()
        .enable_grad()
        .unwrap();
    let source_alias = source.clone();
    let tree = TreeTN::from_tensors(vec![source], vec![0usize]).unwrap();
    let subdomain = SubDomainTreeTN::new(tree, projector([(site.clone(), 1)])).unwrap();
    let masked = subdomain
        .data()
        .tensor(subdomain.data().node_index(&0).unwrap())
        .unwrap();
    assert!(masked.tracks_grad());
    assert_eq!(masked.to_vec::<f64>().unwrap(), vec![0.0, 4.0]);
    masked.sum().unwrap().backward().unwrap();
    assert_eq!(
        source_alias
            .grad()
            .unwrap()
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![0.0, 1.0]
    );

    let diagonal_site = DynIndex::new_dyn(3);
    let diagonal_aux = DynIndex::new_dyn(3);
    let diagonal = IdxTensor::from_diag(
        vec![diagonal_site.clone(), diagonal_aux],
        vec![2.0, 3.0, 5.0],
    )
    .unwrap();
    let diagonal_tree = TreeTN::from_tensors(vec![diagonal], vec![0usize]).unwrap();
    let diagonal_subdomain =
        SubDomainTreeTN::new(diagonal_tree, projector([(diagonal_site, 1)])).unwrap();
    let masked_diagonal = diagonal_subdomain
        .data()
        .tensor(diagonal_subdomain.data().node_index(&0).unwrap())
        .unwrap();
    assert!(masked_diagonal.is_diag());
}

#[test]
fn rejects_invalid_projector_identity_coordinate_and_dtype() {
    let (tree, site, _) = one_node_f64();
    let out_of_range = Projector::from_pairs([(site.clone(), site.dim)]).unwrap_err();
    assert!(matches!(
        out_of_range,
        PartitionedTreeTNError::ProjectorCoordinateOutOfBounds { value, dim, .. }
            if value == dim
    ));

    let tagged = Index::new_with_tags(site.id, site.dim, TagSet::from_str("Site").unwrap());
    let error = SubDomainTreeTN::new(tree.clone(), projector([(tagged.clone(), 0)])).unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::ProjectorIndexNotFound { index } if index == tagged
    ));
    let primed = site.prime();
    let error = SubDomainTreeTN::new(tree.clone(), projector([(primed.clone(), 0)])).unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::ProjectorIndexNotFound { index } if index == primed
    ));

    let larger_same_identity = Index::new_with_tags(site.id, site.dim + 1, site.tags.clone());
    // Same full identity with a different dimension is rejected as a site
    // mismatch before the coordinate is ever checked: the alias must never
    // reach masking.
    let error = SubDomainTreeTN::new(tree, projector([(larger_same_identity.clone(), site.dim)]))
        .unwrap_err();
    assert!(matches!(error, PartitionedTreeTNError::SiteIndexMismatch));

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let mixed = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0_f64, 2.0]).unwrap(),
            IdxTensor::from_dense(
                vec![bond, s1],
                vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            )
            .unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    assert!(matches!(
        SubDomainTreeTN::from_treetn(mixed),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
}

#[test]
fn rejects_invalid_tree_topology() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let b20 = DynIndex::new_dyn(2);
    let mut tree = TreeTN::<IdxTensor, String>::new();
    let a = tree
        .add_tensor(
            "a".to_string(),
            IdxTensor::from_dense(vec![s0, b01.clone(), b20.clone()], vec![1.0; 8]).unwrap(),
        )
        .unwrap();
    let b = tree
        .add_tensor(
            "b".to_string(),
            IdxTensor::from_dense(vec![b01, s1, b12.clone()], vec![1.0; 8]).unwrap(),
        )
        .unwrap();
    let c = tree
        .add_tensor(
            "c".to_string(),
            IdxTensor::from_dense(vec![b12, s2, b20], vec![1.0; 8]).unwrap(),
        )
        .unwrap();
    let a_b01 = tree.tensor(a).unwrap().indices()[1].clone();
    let b_b01 = tree.tensor(b).unwrap().indices()[0].clone();
    let b_b12 = tree.tensor(b).unwrap().indices()[2].clone();
    let c_b12 = tree.tensor(c).unwrap().indices()[0].clone();
    let c_b20 = tree.tensor(c).unwrap().indices()[2].clone();
    let a_b20 = tree.tensor(a).unwrap().indices()[2].clone();
    tree.connect(a, &a_b01, b, &b_b01).unwrap();
    tree.connect(b, &b_b12, c, &c_b12).unwrap();
    tree.connect(c, &c_b20, a, &a_b20).unwrap();

    assert!(matches!(
        SubDomainTreeTN::from_treetn(tree),
        Err(PartitionedTreeTNError::InvalidTopology { .. })
    ));
}

#[test]
fn projector_identity_hash_is_insertion_order_independent_and_full_metadata_matters() {
    let base = DynIndex::new_dyn(2);
    let first = Index::new_with_tags(base.id, 2, TagSet::from_str("Site,Auxiliary").unwrap());
    let second = Index::new_with_tags(base.id, 2, TagSet::from_str("Auxiliary,Site").unwrap());
    let primed = base.prime();
    let a = projector([(first.clone(), 1), (primed.clone(), 0)]);
    let b = projector([(primed.clone(), 0), (second.clone(), 1)]);
    let mut hasher_a = std::collections::hash_map::DefaultHasher::new();
    let mut hasher_b = std::collections::hash_map::DefaultHasher::new();
    a.hash(&mut hasher_a);
    b.hash(&mut hasher_b);
    assert_eq!(a, b);
    assert_eq!(hasher_a.finish(), hasher_b.finish());

    let tagged = projector([(first, 1)]);
    let untagged = projector([(base, 1)]);
    assert_ne!(tagged, untagged);
    let mut map = HashMap::new();
    map.insert(tagged.clone(), 7usize);
    assert_eq!(map.get(&tagged), Some(&7));
    assert!(!map.contains_key(&untagged));
}

#[test]
fn projector_insert_failure_is_transactional() {
    let index = DynIndex::new_dyn(2);
    let mut projector = projector([(index.clone(), 0)]);
    let before = projector.clone();
    assert!(projector.insert(index.clone(), index.dim).is_err());
    assert_eq!(projector, before);
}

#[test]
fn from_treetn_rebuilds_without_storing_projector_metadata_as_canonical_state() {
    let (tree, site0, _) = one_node_f64();
    let mut tree = tree;
    tree.set_canonical_region(["root".to_string()]).unwrap();
    let subdomain = SubDomainTreeTN::from_treetn(tree).unwrap();
    assert!(subdomain.projector().is_empty());
    assert!(subdomain.data().canonical_region().is_empty());
    assert_eq!(subdomain.max_bond_dim(), 1);
    assert!(!subdomain.is_empty());
    assert!(subdomain.all_indices().contains(&site0));
}

#[test]
fn subdomain_add_is_strict_and_preserves_full_site_axes() {
    let (tree, site0, site1) = one_node_f64();
    let projector = projector([(site0.clone(), 0)]);
    let left = SubDomainTreeTN::new(tree.clone(), projector.clone()).unwrap();
    let right = SubDomainTreeTN::new(tree, projector).unwrap();

    let sum = left.add(&right).unwrap();
    let tensor = sum
        .data()
        .tensor(sum.data().node_index(&"root".to_string()).unwrap())
        .unwrap();
    assert_eq!(tensor.indices().len(), 2);
    assert_eq!(tensor.indices(), &[site0, site1]);
    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![2.0, 0.0, 6.0, 0.0]);
}

#[test]
fn subdomain_truncate_uses_explicit_center_and_is_transactional_on_validation_errors() {
    let (tree, sites) = chain_f64();
    let mut subdomain = SubDomainTreeTN::new(tree, projector([(sites[0].clone(), 0)])).unwrap();
    let before_indices = subdomain.all_indices();
    let before_projector = subdomain.projector().clone();

    subdomain
        .truncate(
            &"middle".to_string(),
            TruncationOptions::default().with_max_bond_dim(1),
        )
        .unwrap();
    assert!(subdomain.max_bond_dim() <= 1);
    assert_eq!(subdomain.all_indices(), before_indices);
    assert_eq!(subdomain.projector(), &before_projector);

    let before = subdomain
        .data()
        .tensor(subdomain.data().node_index(&"middle".to_string()).unwrap())
        .unwrap()
        .clone();
    assert!(matches!(
        subdomain.truncate(&"missing".to_string(), TruncationOptions::default()),
        Err(PartitionedTreeTNError::InvalidCenter)
    ));
    assert!(matches!(
        subdomain.truncate(
            &"middle".to_string(),
            TruncationOptions::default().with_max_bond_dim(0),
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    let after = subdomain
        .data()
        .tensor(subdomain.data().node_index(&"middle".to_string()).unwrap())
        .unwrap();
    assert_eq!(after.indices(), before.indices());
    assert_eq!(
        after.to_vec::<f64>().unwrap(),
        before.to_vec::<f64>().unwrap()
    );
}

#[test]
fn subdomain_contraction_rejects_site_indices_assigned_to_different_nodes() {
    let shared = DynIndex::new_dyn(2);
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let left_bond = DynIndex::new_dyn(1);
    let right_bond = DynIndex::new_dyn(1);
    let left = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![shared.clone(), left_bond.clone()], vec![1.0, 1.0]).unwrap(),
            IdxTensor::from_dense(vec![left_bond, left_site], vec![1.0, 1.0]).unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let right = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![right_bond.clone(), right_site], vec![1.0, 1.0]).unwrap(),
            IdxTensor::from_dense(vec![shared, right_bond], vec![1.0, 1.0]).unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let left = SubDomainTreeTN::from_treetn(left).unwrap();
    let right = SubDomainTreeTN::from_treetn(right).unwrap();

    assert!(matches!(
        left.contract(&right, &0, ContractionOptions::default()),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));
}

#[test]
fn long_chain_norm_avoids_full_dense_materialization() {
    let node_count = 32usize;
    let sites: Vec<_> = (0..node_count).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (0..node_count - 1).map(|_| DynIndex::new_dyn(1)).collect();
    let mut tensors = Vec::with_capacity(node_count);
    for node in 0..node_count {
        let tensor = if node == 0 {
            IdxTensor::from_dense(vec![sites[node].clone(), bonds[0].clone()], vec![1.0, 1.0])
                .unwrap()
        } else if node + 1 == node_count {
            IdxTensor::from_dense(
                vec![bonds[node - 1].clone(), sites[node].clone()],
                vec![1.0, 1.0],
            )
            .unwrap()
        } else {
            IdxTensor::from_dense(
                vec![
                    bonds[node - 1].clone(),
                    sites[node].clone(),
                    bonds[node].clone(),
                ],
                vec![1.0, 1.0],
            )
            .unwrap()
        };
        tensors.push(tensor);
    }
    let tree = TreeTN::from_tensors(tensors, (0..node_count).collect()).unwrap();
    let subdomain = SubDomainTreeTN::new(tree, projector([(sites[0].clone(), 0)])).unwrap();

    let expected = 2.0_f64.powi((node_count - 1) as i32);
    let actual = subdomain.norm_squared().unwrap();
    assert!((actual - expected).abs() / expected < 1.0e-12);
}
