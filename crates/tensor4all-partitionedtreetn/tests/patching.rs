use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, SvdTruncationPolicy};
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

/// Materialize `source` and `result` densely once and return `maxabs(diff)`.
///
/// Follows the repository dense-comparison policy: subtract the materialized
/// tensors and report the largest absolute entry; never re-materialize within
/// the comparison.
fn dense_maxabs_diff(source: &SubDomainTreeTN, result: &PartitionedTreeTN) -> f64 {
    let original = source
        .data()
        .clone()
        .to_dense()
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let truncated = result
        .to_treetn()
        .unwrap()
        .to_dense()
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    original
        .iter()
        .zip(&truncated)
        .fold(0.0_f64, |max_abs, (x, y)| max_abs.max((x - y).abs()))
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
    // squared weight `a*a` on every internal bond. The local discarded-weight
    // cutoff is best effort: each bond's whole local cutoff may trim that
    // mode, and no test here asserts a whole-network error bound.
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

/// The two-site chain `|00> + a|11>` has squared Schmidt weights `[1, a*a]`
/// on a system squared norm `1 + a*a`, so the whole local cutoff equals
/// `cutoff * (1 + a*a)` applied at the (single effective) local SVD.
#[test]
fn local_cutoff_truncates_known_spectrum_with_exact_ranks() {
    let a = 0.1;
    let patch = two_site_chain(a);
    let partition = PartitionedTreeTN::from_subdomain(patch.clone()).unwrap();

    // cutoff well below the second squared weight keeps rank 2 and the result
    // stays essentially exact.
    let below = truncate_adaptive(&partition, &0, 1.0e-8, None).unwrap();
    assert_eq!(below.values().next().unwrap().max_bond_dim(), 2);
    assert!(dense_maxabs_diff(&patch, &below) < 1.0e-9);

    // cutoff above the second squared weight trims to rank 1, discarding only
    // the small mode `a|11>` (weight a*a = 0.01).
    let above = truncate_adaptive(&partition, &0, 1.0e-2, None).unwrap();
    assert_eq!(above.values().next().unwrap().max_bond_dim(), 1);
    assert!(dense_maxabs_diff(&patch, &above) < 2.1e-1);

    // cutoff == 0 disables threshold truncation entirely.
    let exact = truncate_adaptive(&partition, &0, 0.0, None).unwrap();
    assert_eq!(exact.values().next().unwrap().max_bond_dim(), 2);
}

#[test]
fn local_cutoff_hard_cap_takes_precedence_in_both_directions() {
    let a = 0.1;
    let partition = PartitionedTreeTN::from_subdomain(two_site_chain(a)).unwrap();

    // cutoff == 0 leaves thresholds off; the hard cap still trims the rank.
    let capped = truncate_adaptive(&partition, &0, 0.0, Some(1)).unwrap();
    assert_eq!(capped.values().next().unwrap().max_bond_dim(), 1);

    // cutoff above the boundary trims even without a cap (and does not drop
    // the patch because its norm exceeds the local cutoff).
    let uncapped = truncate_adaptive(&partition, &0, 0.5, None).unwrap();
    assert_eq!(uncapped.len(), 1);
    assert_eq!(uncapped.values().next().unwrap().max_bond_dim(), 1);
}

#[test]
fn truncate_adaptive_local_cutoff_applies_whole_threshold_on_multi_edge_patches() {
    // A single unprojected 4-site patch with an independent small Schmidt mode
    // on every bond. The whole local cutoff is reused at each local SVD; reuse
    // is deliberate (no per-edge split), so we assert the structural behavior
    // (the small modes are dropped) and explicitly do not assert a
    // whole-network error bound.
    let patch = four_site_chain(0.095);
    let partition = PartitionedTreeTN::from_subdomain(patch).unwrap();
    let result = truncate_adaptive(&partition, &1, 0.1, None).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.values().next().unwrap().max_bond_dim(), 1);

    // A tight cutoff keeps the small modes (no threshold truncation, no cap).
    let kept = truncate_adaptive(&partition, &1, 0.0, None).unwrap();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept.values().next().unwrap().max_bond_dim(), 2);
}

#[test]
fn local_cutoff_patch_drop_follows_the_drop_boundary() {
    let site = DynIndex::new_dyn(2);
    // eager masking keeps only the projected coordinate, so heavy (coord 0)
    // has norm^2 = 9 and light (coord 1) has norm^2 = 1.
    let heavy = one_site_patch(&site, [3.0, 0.0], 0);
    let light = one_site_patch(&site, [0.0, 1.0], 1);
    let partition = PartitionedTreeTN::from_subdomains(vec![heavy, light]).unwrap();
    // total norm^2 = 10; both projected single-site patches have volume 1 so
    // total volume = 2 and each local cutoff = cutoff * 10 * (1/2) = 5*cutoff.

    // cutoff 0.2 -> local 1.0: the light patch (1 <= 1.0) is dropped.
    let dropped = truncate_adaptive(&partition, &0, 0.2, None).unwrap();
    assert_eq!(dropped.len(), 1);

    // cutoff 0.1 -> local 0.5: both patches (1 > 0.5, 9 > 0.5) are kept.
    let kept = truncate_adaptive(&partition, &0, 0.1, None).unwrap();
    assert_eq!(kept.len(), 2);
}

#[test]
fn local_cutoff_truncates_complex64_systems() {
    // |00> + (0.1 + 0.2i)|11> has squared Schmidt weights [1, |z|^2 = 0.05];
    // a cutoff above 0.05 trims to rank 1, below keeps rank 2.
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let z = Complex64::new(0.1, 0.2);
    let t0 = IdxTensor::from_dense(
        vec![s0, bond.clone()],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            z,
        ],
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![bond, s1],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            z,
        ],
    )
    .unwrap();
    let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1]).unwrap();
    let partition =
        PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(tree).unwrap()).unwrap();

    let kept = truncate_adaptive(&partition, &0, 1.0e-4, None).unwrap();
    assert_eq!(kept.values().next().unwrap().max_bond_dim(), 2);
    let truncated = truncate_adaptive(&partition, &0, 1.0e-1, None).unwrap();
    assert_eq!(truncated.values().next().unwrap().max_bond_dim(), 1);
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
            cutoff: 1.0e-24,
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
fn add_with_patching_splits_over_cap_without_claiming_a_global_error_bound() {
    // Run a multi-edge patch through the forced-split path. We assert only
    // structural and repeatability properties: the local `cutoff` is best
    // effort and no test here asserts a whole-network error bound.
    let patch = four_site_chain(0.09);
    // A tight cutoff leaves the small Schmidt modes (squared weight ~0.008)
    // intact, so the hard cap drives the split path.
    let options = PatchingOptions {
        cutoff: 1.0e-8,
        max_bond_dim: Some(1),
        ..PatchingOptions::default()
    };
    let result = add_with_patching(vec![patch.clone()], &1, &options).unwrap();
    assert!(result.len() > 1);
    assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));

    // Re-running the same input is deterministic (same layout/ranks).
    let rerun = add_with_patching(vec![patch], &1, &options).unwrap();
    assert_eq!(result.len(), rerun.len());
    assert!(result
        .projectors()
        .all(|projector| rerun.contains(projector)));
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
        cutoff: 0.0,
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
        cutoff: 0.0,
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
        cutoff: 0.0,
        max_bond_dim: Some(1),
        ..PatchingOptions::default()
    };

    let result =
        contract_adaptive(&left, &right, &0, &ContractionOptions::default(), &options).unwrap();

    assert_eq!(result.len(), 1);
    assert!(result.values().next().unwrap().all_indices().is_empty());
}

#[test]
fn contract_duplicate_output_groups_are_order_independent() {
    // Three disjoint one-site patches on each side contract to the same scalar
    // output projector. The duplicate contributions must be exact-added in a
    // deterministic order and truncated as one completed group, so reversing
    // the insertion order yields an identical result.
    let site = DynIndex::new_dyn(3);
    let patch_fn = |coordinate: usize, value: f64| {
        let mut values = vec![0.0_f64; 3];
        values[coordinate] = value;
        SubDomainTreeTN::new(
            TreeTN::from_tensors(
                vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
                vec![0usize],
            )
            .unwrap(),
            Projector::from_pairs([(site.clone(), coordinate)]).unwrap(),
        )
        .unwrap()
    };
    let left = PartitionedTreeTN::from_subdomains(vec![
        patch_fn(0, 1.0),
        patch_fn(1, 2.0),
        patch_fn(2, 3.0),
    ])
    .unwrap();
    let right = PartitionedTreeTN::from_subdomains(vec![
        patch_fn(0, 10.0),
        patch_fn(1, 20.0),
        patch_fn(2, 30.0),
    ])
    .unwrap();
    let left_reversed = PartitionedTreeTN::from_subdomains(vec![
        patch_fn(2, 3.0),
        patch_fn(1, 2.0),
        patch_fn(0, 1.0),
    ])
    .unwrap();
    let right_reversed = PartitionedTreeTN::from_subdomains(vec![
        patch_fn(2, 30.0),
        patch_fn(0, 10.0),
        patch_fn(1, 20.0),
    ])
    .unwrap();

    let forward = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    let reversed = left_reversed
        .contract(&right_reversed, &0, ContractionOptions::default())
        .unwrap();

    // Identical layout and ranks, and identical dense values (1*10 + 2*20 +
    // 3*30 = 140 in deterministic order).
    assert_eq!(forward.len(), 1);
    assert_eq!(reversed.len(), 1);
    assert!((forward.norm().unwrap() - 140.0).abs() < 1.0e-9);
    assert!((reversed.norm().unwrap() - 140.0).abs() < 1.0e-9);
    let forward_vec = forward
        .to_treetn()
        .unwrap()
        .to_dense()
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let reversed_vec = reversed
        .to_treetn()
        .unwrap()
        .to_dense()
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(forward_vec, reversed_vec);
}

#[test]
fn adaptive_paths_reject_invalid_options_before_shortcut_and_ordinary_paths() {
    let site = DynIndex::new_dyn(2);
    let patch = one_site_patch(&site, [1.0, 2.0], 0);
    let partition = PartitionedTreeTN::from_subdomain(patch.clone()).unwrap();
    let empty = PartitionedTreeTN::<usize>::new();
    let bad_cutoffs = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0];

    // `truncate_adaptive` rejects every bad cutoff on both the ordinary and
    // the empty-partition shortcut path.
    for cutoff in bad_cutoffs {
        assert!(
            matches!(
                truncate_adaptive(&partition, &0, cutoff, None),
                Err(PartitionedTreeTNError::InvalidOptions { .. })
            ),
            "truncate_adaptive must reject cutoff {cutoff} on a populated partition"
        );
        assert!(
            matches!(
                truncate_adaptive(&empty, &0, cutoff, None),
                Err(PartitionedTreeTNError::InvalidOptions { .. })
            ),
            "truncate_adaptive must reject cutoff {cutoff} on the empty shortcut"
        );
    }
    assert!(matches!(
        truncate_adaptive(&empty, &0, 0.0, Some(0)),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));

    // `add_with_patching` rejects bad cutoffs and the zero cap without
    // mutating its inputs.
    let before = partition.clone();
    for cutoff in bad_cutoffs {
        assert!(
            matches!(
                add_with_patching(
                    vec![patch.clone()],
                    &0,
                    &PatchingOptions {
                        cutoff,
                        ..PatchingOptions::default()
                    },
                ),
                Err(PartitionedTreeTNError::InvalidOptions { .. })
            ),
            "add_with_patching must reject cutoff {cutoff}"
        );
    }
    assert!(matches!(
        add_with_patching(
            vec![patch.clone()],
            &0,
            &PatchingOptions {
                max_bond_dim: Some(0),
                ..PatchingOptions::default()
            },
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert_eq!(partition.len(), before.len());

    // `contract_adaptive` validates contraction options before the
    // empty-operand shortcut, and rejects bad patching cutoffs.
    assert!(matches!(
        contract_adaptive(
            &PartitionedTreeTN::new(),
            &PartitionedTreeTN::new(),
            &0,
            &ContractionOptions::default().with_svd_policy(SvdTruncationPolicy::new(f64::NAN)),
            &PatchingOptions::default(),
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        contract_adaptive(
            &partition,
            &partition,
            &0,
            &ContractionOptions::default(),
            &PatchingOptions {
                cutoff: f64::NAN,
                ..PatchingOptions::default()
            },
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        contract_adaptive(
            &partition,
            &partition,
            &0,
            &ContractionOptions::default().with_max_bond_dim(0),
            &PatchingOptions::default(),
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
}
