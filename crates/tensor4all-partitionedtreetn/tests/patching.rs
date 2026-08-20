use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    add_with_patching, contract_adaptive, truncate_adaptive, PartitionedTreeTN,
    PartitionedTreeTNError, PatchSplitStrategy, PatchingOptions, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};

fn rank_two_chain() -> (SubDomainTreeTN, DynIndex, DynIndex) {
    let site0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let site1 = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![site0.clone(), bond.clone()],
        vec![1.0_f64, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let right =
        IdxTensor::from_dense(vec![bond, site1.clone()], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).unwrap();
    (SubDomainTreeTN::from_treetn(tree).unwrap(), site0, site1)
}

fn one_site_patch(site: &DynIndex, values: [f64; 2], coordinate: usize) -> SubDomainTreeTN {
    let tensor = IdxTensor::from_dense(vec![site.clone()], values.to_vec()).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    SubDomainTreeTN::new(
        tree,
        Projector::from_pairs([(site.clone(), coordinate)]).unwrap(),
    )
    .unwrap()
}

fn branched_complex_tree() -> (TreeTN<IdxTensor, String>, Vec<DynIndex>) {
    let center_sites = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(3)];
    let leaf_sites = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let bonds = vec![DynIndex::new_dyn(1), DynIndex::new_dyn(1)];
    let one = Complex64::new(1.0, 1.0);
    let center = IdxTensor::from_dense(
        vec![
            center_sites[0].clone(),
            center_sites[1].clone(),
            bonds[0].clone(),
            bonds[1].clone(),
        ],
        vec![one; 6],
    )
    .unwrap();
    let leaves = leaf_sites
        .iter()
        .zip(&bonds)
        .map(|(site, bond)| {
            IdxTensor::from_dense(vec![bond.clone(), site.clone()], vec![one; 2]).unwrap()
        })
        .collect::<Vec<_>>();
    let mut tensors = vec![center];
    tensors.extend(leaves);
    let tree = TreeTN::from_tensors(
        tensors,
        vec![
            "center".to_string(),
            "leaf0".to_string(),
            "leaf1".to_string(),
        ],
    )
    .unwrap();
    (tree, center_sites.into_iter().chain(leaf_sites).collect())
}

/// Materialize `patch` and `result` densely and report the Frobenius relative error.
fn dense_relative_error(source: &SubDomainTreeTN, result: &PartitionedTreeTN) -> f64 {
    let original = source.data().clone().to_dense().unwrap();
    let original_vec: Vec<f64> = original.to_vec().unwrap();
    let truncated = result.to_treetn().unwrap().to_dense().unwrap();
    let truncated_vec: Vec<f64> = truncated.to_vec().unwrap();
    let diff_squared: f64 = original_vec
        .iter()
        .zip(&truncated_vec)
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    let norm_squared: f64 = original_vec.iter().map(|x| x * x).sum();
    (diff_squared / norm_squared).sqrt()
}

/// Build an unprojected 4-site chain state
/// `|0000> + a(|1000> + |0011> + |0001>)` with one small Schmidt mode on
/// every internal bond.
fn four_site_chain(a: f64) -> SubDomainTreeTN {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let b23 = DynIndex::new_dyn(2);
    // The state `|0000> + a|1000> + a|0111>` has one small Schmidt mode of
    // squared weight `a*a` on every internal bond. Before the per-bond budget
    // split, each local SVD reused the full patch budget and the accumulated
    // squared error exceeded `rtol^2 * ||F||^2`.
    let t0 = IdxTensor::from_dense(vec![s0, b01.clone()], vec![1.0, 0.0, 0.0, a]).unwrap();
    let t1 = IdxTensor::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![b12, s2.clone(), b23.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a],
    )
    .unwrap();
    let t3 = IdxTensor::from_dense(vec![b23, s3], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let tree = TreeTN::from_tensors(vec![t0, t1, t2, t3], vec![0usize, 1, 2, 3]).unwrap();
    SubDomainTreeTN::from_treetn(tree).unwrap()
}

/// Build an unprojected 2-site GHZ-like chain `|00> + a|11>`.
fn two_site_chain(a: f64) -> SubDomainTreeTN {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0, 0.0, a]).unwrap();
    let t1 = IdxTensor::from_dense(vec![bond, s1], vec![1.0, 0.0, 0.0, a]).unwrap();
    let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1]).unwrap();
    SubDomainTreeTN::from_treetn(tree).unwrap()
}

#[test]
fn truncate_adaptive_global_error_stays_within_rtol_on_multi_edge_patches() {
    // A single unprojected 4-site patch with an independent small Schmidt mode
    // on every bond. Before the per-bond budget split, each local SVD reused
    // the full patch budget and the accumulated error exceeded rtol.
    let patch = four_site_chain(0.095);
    let partition = PartitionedTreeTN::from_subdomain(patch.clone()).unwrap();
    let rtol = 0.1;

    let result = truncate_adaptive(&partition, &1, rtol, None).unwrap();
    assert_eq!(result.len(), 1);
    let relative_error = dense_relative_error(&patch, &result);
    assert!(
        relative_error <= rtol,
        "global relative error {relative_error} exceeds rtol {rtol}"
    );

    // The same guarantee holds for a single-bond patch, which must still be
    // truncatable within its full (unsplit) budget.
    let compact = two_site_chain(0.1);
    let compact_partition = PartitionedTreeTN::from_subdomain(compact.clone()).unwrap();
    let compact_result = truncate_adaptive(&compact_partition, &0, rtol, None).unwrap();
    let compact_error = dense_relative_error(&compact, &compact_result);
    assert!(
        compact_error <= rtol,
        "single-bond relative error {compact_error} exceeds rtol {rtol}"
    );
    assert!(compact_result.values().next().unwrap().max_bond_dim() == 1);
}

#[test]
fn add_with_patching_sums_equal_key_inputs_instead_of_replacing_them() {
    let site = DynIndex::new_dyn(2);
    let make = |values| {
        SubDomainTreeTN::from_treetn(
            TreeTN::from_tensors(
                vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
                vec![0usize],
            )
            .unwrap(),
        )
        .unwrap()
    };
    // Two unprojected patches over the same site: `add_with_patching` must
    // produce the summed patch [1, 1], not the silent last-write-wins [0, 1].
    let result = add_with_patching(
        vec![make(vec![0.0, 1.0_f64]), make(vec![1.0, 0.0])],
        &0,
        &PatchingOptions {
            rtol: 1.0e-12,
            max_bond_dim: Some(2),
            ..PatchingOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    let patch = result.values().next().unwrap();
    let tensor = patch
        .data()
        .tensor(patch.data().node_index(&0).unwrap())
        .unwrap();
    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn add_with_patching_global_error_stays_within_rtol_with_repeated_truncation() {
    // Run the multi-edge patch through the split path (cap forces site
    // splitting) and verify the final global error is still within rtol.
    let patch = four_site_chain(0.09);
    let rtol = 0.1;
    let result = add_with_patching(
        vec![patch.clone()],
        &1,
        &PatchingOptions {
            rtol,
            max_bond_dim: Some(1),
            ..PatchingOptions::default()
        },
    )
    .unwrap();
    assert!(result.len() > 1);
    let relative_error = dense_relative_error(&patch, &result);
    assert!(
        relative_error <= rtol,
        "global relative error {relative_error} exceeds rtol {rtol}"
    );
}

fn assert_branched_complex_patch(patch: &SubDomainTreeTN<String>, sites: &[DynIndex]) {
    assert_eq!(patch.node_count(), 3);
    assert_eq!(patch.all_indices().len(), sites.len());
    assert!(sites.iter().all(|site| patch.all_indices().contains(site)));
    for name in patch.data().node_names() {
        let tensor = patch
            .data()
            .tensor(patch.data().node_index(&name).unwrap())
            .unwrap();
        assert!(tensor.is_c64());
        assert!(!tensor.to_vec::<Complex64>().unwrap().is_empty());
    }
    assert!((patch.norm_squared().unwrap() - 32.0).abs() < 1.0e-10);
}

fn assert_branched_complex_partition(partition: &PartitionedTreeTN<String>, sites: &[DynIndex]) {
    assert_eq!(partition.len(), 6);
    assert!((partition.norm_squared().unwrap() - 192.0).abs() < 1.0e-10);
    for patch in partition.values() {
        assert_eq!(patch.max_bond_dim(), 1);
        assert_branched_complex_patch(patch, sites);
    }
}

#[test]
fn adaptive_patching_preserves_branched_complex_multisite_tree() {
    let (tree, sites) = branched_complex_tree();
    let patch = SubDomainTreeTN::from_treetn(tree).unwrap();
    let center = "center".to_string();
    let options = PatchingOptions {
        rtol: 0.0,
        max_bond_dim: Some(1),
        patch_order: sites[..2].to_vec(),
        split_strategy: PatchSplitStrategy::Sequential,
    };

    let patched = add_with_patching(vec![patch], &center, &options).unwrap();
    assert_branched_complex_partition(&patched, &sites);

    let retruncated = truncate_adaptive(&patched, &center, 0.0, Some(1)).unwrap();
    assert_branched_complex_partition(&retruncated, &sites);
}

#[test]
fn add_with_patching_splits_at_explicit_center() {
    let (subdomain, site0, _) = rank_two_chain();
    let options = PatchingOptions {
        rtol: 0.0,
        max_bond_dim: Some(1),
        patch_order: vec![site0.clone()],
        split_strategy: PatchSplitStrategy::Sequential,
    };

    let result = add_with_patching(vec![subdomain], &0, &options).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
    assert!(result
        .projectors()
        .all(|projector| projector.get(&site0).is_some()));
}

#[test]
fn truncate_adaptive_uses_absolute_volume_budget_and_keeps_eager_values() {
    let site = DynIndex::new_dyn(2);
    let high = one_site_patch(&site, [10.0, 1.0e12], 0);
    let low = one_site_patch(&site, [1.0e12, 0.01], 1);
    let partition = PartitionedTreeTN::from_subdomains(vec![high, low]).unwrap();

    let result = truncate_adaptive(&partition, &0, 0.1, Some(2)).unwrap();

    assert_eq!(result.len(), 1);
    let projector = Projector::from_pairs([(site, 0)]).unwrap();
    assert!(result.contains(&projector));
    assert!((result.get(&projector).unwrap().norm_squared().unwrap() - 100.0).abs() < 1e-10);
}

#[test]
fn contract_adaptive_retruncates_a_new_partition() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(
        SubDomainTreeTN::from_treetn(
            TreeTN::from_tensors(
                vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0]).unwrap()],
                vec![0usize],
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let right = PartitionedTreeTN::from_subdomain(
        SubDomainTreeTN::from_treetn(
            TreeTN::from_tensors(
                vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0]).unwrap()],
                vec![0usize],
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let options = PatchingOptions {
        rtol: 0.0,
        max_bond_dim: Some(1),
        ..PatchingOptions::default()
    };

    let result =
        contract_adaptive(&left, &right, &0, &ContractionOptions::default(), &options).unwrap();

    assert_eq!(result.len(), 1);
    assert!(result.values().next().unwrap().all_indices().is_empty());
}

#[test]
fn adaptive_paths_validate_before_shortcuts_and_are_transactional() {
    let site = DynIndex::new_dyn(2);
    let partition =
        PartitionedTreeTN::from_subdomain(one_site_patch(&site, [1.0, 2.0], 0)).unwrap();
    let before = partition.clone();

    assert!(matches!(
        truncate_adaptive(&partition, &99, 0.0, Some(1)),
        Err(PartitionedTreeTNError::InvalidCenter)
    ));
    assert!(matches!(
        truncate_adaptive(&partition, &0, f64::NAN, Some(1)),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        add_with_patching(
            vec![partition.values().next().unwrap().clone()],
            &99,
            &PatchingOptions {
                max_bond_dim: Some(0),
                ..PatchingOptions::default()
            },
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert_eq!(partition.len(), before.len());
}
