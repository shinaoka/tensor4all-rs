use super::*;
use crate::random::{random_treetn, LinkSpace};
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::IdxTensor;

type DynIndex = Index<DynId, TagSet>;

fn make_index(dim: usize) -> DynIndex {
    Index::new_dyn(dim)
}

fn create_chain_site_network(n: usize) -> SiteIndexNetwork<String, DynIndex> {
    let mut net = SiteIndexNetwork::new();
    for i in 0..n {
        let name = format!("N{}", i);
        let site_idx = make_index(2);
        net.add_node(name, [site_idx].into_iter().collect::<HashSet<_>>())
            .unwrap();
    }
    for i in 0..(n - 1) {
        net.add_edge(&format!("N{}", i), &format!("N{}", i + 1))
            .unwrap();
    }
    net
}

/// Create a simple LinearOperator from a TreeTN with explicit index mappings.
fn create_linear_operator_from_treetn(
    mpo: TreeTN<IdxTensor, String>,
    input_indices: &[(String, DynIndex, DynIndex)], // (node, true_input, internal_input)
    output_indices: &[(String, DynIndex, DynIndex)], // (node, true_output, internal_output)
) -> LinearOperator<IdxTensor, String> {
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for (node, true_idx, internal_idx) in input_indices {
        input_mapping.insert(
            node.clone(),
            IndexMapping {
                true_index: true_idx.clone(),
                internal_index: internal_idx.clone(),
            },
        );
    }

    for (node, true_idx, internal_idx) in output_indices {
        output_mapping.insert(
            node.clone(),
            IndexMapping {
                true_index: true_idx.clone(),
                internal_index: internal_idx.clone(),
            },
        );
    }

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

#[test]
fn test_are_exclusive_disjoint() {
    // Target: N0 -- N1 -- N2 -- N3 -- N4
    let target = create_chain_site_network(5);

    // Create two non-overlapping operators
    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    op1_net.add_node("N0".to_string(), HashSet::new()).unwrap();
    op1_net.add_node("N1".to_string(), HashSet::new()).unwrap();
    op1_net
        .add_edge(&"N0".to_string(), &"N1".to_string())
        .unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    op2_net.add_node("N3".to_string(), HashSet::new()).unwrap();
    op2_net.add_node("N4".to_string(), HashSet::new()).unwrap();
    op2_net
        .add_edge(&"N3".to_string(), &"N4".to_string())
        .unwrap();

    // Create TreeTNs with these networks
    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let op1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let op2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    // Test exclusivity
    let result = are_exclusive_operators::<IdxTensor, _, _>(&target, &[&op1, &op2]);
    assert!(result, "Disjoint operators should be exclusive");
}

#[test]
fn test_are_exclusive_overlapping() {
    // Target: N0 -- N1 -- N2 -- N3
    let target = create_chain_site_network(4);

    // Create overlapping operators (both include N1)
    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    op1_net.add_node("N0".to_string(), HashSet::new()).unwrap();
    op1_net.add_node("N1".to_string(), HashSet::new()).unwrap();
    op1_net
        .add_edge(&"N0".to_string(), &"N1".to_string())
        .unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    op2_net.add_node("N1".to_string(), HashSet::new()).unwrap();
    op2_net.add_node("N2".to_string(), HashSet::new()).unwrap();
    op2_net
        .add_edge(&"N1".to_string(), &"N2".to_string())
        .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let op1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let op2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    let result = are_exclusive_operators::<IdxTensor, _, _>(&target, &[&op1, &op2]);
    assert!(!result, "Overlapping operators should not be exclusive");
}

#[test]
fn test_are_exclusive_single_node_operators() {
    // Target: N0 -- N1 -- N2 -- N3
    let target = create_chain_site_network(4);

    // Create single-node operators
    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    op1_net.add_node("N0".to_string(), HashSet::new()).unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    op2_net.add_node("N2".to_string(), HashSet::new()).unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let op1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let op2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    let result = are_exclusive_operators::<IdxTensor, _, _>(&target, &[&op1, &op2]);
    assert!(result, "Single-node disjoint operators should be exclusive");
}

// =========================================================================
// Integration tests for compose_exclusive_linear_operators
// =========================================================================

#[test]
fn test_compose_exclusive_linear_operators_basic() {
    // Target: N0 -- N1 -- N2 -- N3 -- N4
    // Op1 covers N0, N1
    // Op2 covers N3, N4
    // Gap: N2 (needs identity)
    let target = create_chain_site_network(5);

    // Create site networks for operators
    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    op1_net
        .add_node(
            "N0".to_string(),
            [s0_in.clone(), s0_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();
    op1_net
        .add_node(
            "N1".to_string(),
            [s1_in.clone(), s1_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();
    op1_net
        .add_edge(&"N0".to_string(), &"N1".to_string())
        .unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s3_in = make_index(2);
    let s3_out = make_index(2);
    let s4_in = make_index(2);
    let s4_out = make_index(2);
    op2_net
        .add_node(
            "N3".to_string(),
            [s3_in.clone(), s3_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();
    op2_net
        .add_node(
            "N4".to_string(),
            [s4_in.clone(), s4_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();
    op2_net
        .add_edge(&"N3".to_string(), &"N4".to_string())
        .unwrap();

    // Create TreeTNs for the operators
    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let mpo2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    // True site indices (what the composed operator maps from/to)
    let true_s0 = make_index(2);
    let true_s1 = make_index(2);
    let true_s3 = make_index(2);
    let true_s4 = make_index(2);

    // Create LinearOperators with explicit mappings
    let lin_op1 = create_linear_operator_from_treetn(
        mpo1,
        &[
            ("N0".to_string(), true_s0.clone(), s0_in.clone()),
            ("N1".to_string(), true_s1.clone(), s1_in.clone()),
        ],
        &[
            ("N0".to_string(), true_s0.clone(), s0_out.clone()),
            ("N1".to_string(), true_s1.clone(), s1_out.clone()),
        ],
    );

    let lin_op2 = create_linear_operator_from_treetn(
        mpo2,
        &[
            ("N3".to_string(), true_s3.clone(), s3_in.clone()),
            ("N4".to_string(), true_s4.clone(), s4_in.clone()),
        ],
        &[
            ("N3".to_string(), true_s3.clone(), s3_out.clone()),
            ("N4".to_string(), true_s4.clone(), s4_out.clone()),
        ],
    );

    // Gap site indices for N2
    let true_s2_in = make_index(2);
    let true_s2_out = make_index(2);
    let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
    gap_site_indices.insert(
        "N2".to_string(),
        vec![(true_s2_in.clone(), true_s2_out.clone())],
    );

    // Compose the operators
    let composed =
        compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
            .expect("Composition should succeed");

    // Verify the composed operator
    let node_names = composed.node_names();
    assert_eq!(node_names.len(), 5, "Composed operator should have 5 nodes");
    assert!(node_names.contains("N0"));
    assert!(node_names.contains("N1"));
    assert!(node_names.contains("N2")); // Gap node
    assert!(node_names.contains("N3"));
    assert!(node_names.contains("N4"));

    // Verify mappings exist for all nodes
    assert!(composed.get_input_mapping(&"N0".to_string()).is_some());
    assert!(composed.get_input_mapping(&"N1".to_string()).is_some());
    assert!(composed.get_input_mapping(&"N2".to_string()).is_some()); // Gap
    assert!(composed.get_input_mapping(&"N3".to_string()).is_some());
    assert!(composed.get_input_mapping(&"N4".to_string()).is_some());

    assert!(composed.get_output_mapping(&"N0".to_string()).is_some());
    assert!(composed.get_output_mapping(&"N1".to_string()).is_some());
    assert!(composed.get_output_mapping(&"N2".to_string()).is_some()); // Gap
    assert!(composed.get_output_mapping(&"N3".to_string()).is_some());
    assert!(composed.get_output_mapping(&"N4".to_string()).is_some());
}

#[test]
fn test_compose_exclusive_linear_operators_single_operators() {
    // Target: N0 -- N1 -- N2
    // Op1 covers N0
    // Op2 covers N2
    // Gap: N1 (needs identity)
    let target = create_chain_site_network(3);

    // Create single-node site networks
    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    op1_net
        .add_node(
            "N0".to_string(),
            [s0_in.clone(), s0_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s2_in = make_index(2);
    let s2_out = make_index(2);
    op2_net
        .add_node(
            "N2".to_string(),
            [s2_in.clone(), s2_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();

    // Create TreeTNs
    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let mpo2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    // True site indices
    let true_s0 = make_index(2);
    let true_s2 = make_index(2);

    // Create LinearOperators
    let lin_op1 = create_linear_operator_from_treetn(
        mpo1,
        &[("N0".to_string(), true_s0.clone(), s0_in.clone())],
        &[("N0".to_string(), true_s0.clone(), s0_out.clone())],
    );

    let lin_op2 = create_linear_operator_from_treetn(
        mpo2,
        &[("N2".to_string(), true_s2.clone(), s2_in.clone())],
        &[("N2".to_string(), true_s2.clone(), s2_out.clone())],
    );

    // Gap for N1
    let true_s1_in = make_index(2);
    let true_s1_out = make_index(2);
    let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
    gap_site_indices.insert(
        "N1".to_string(),
        vec![(true_s1_in.clone(), true_s1_out.clone())],
    );

    // Compose
    let composed =
        compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
            .expect("Composition should succeed");

    // Verify
    assert_eq!(composed.node_names().len(), 3);
}

#[test]
fn test_compose_exclusive_linear_operators_no_gap() {
    // Target: N0 -- N1
    // Op1 covers N0
    // Op2 covers N1
    // No gap
    let target = create_chain_site_network(2);

    // Create single-node site networks
    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    op1_net
        .add_node(
            "N0".to_string(),
            [s0_in.clone(), s0_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    op2_net
        .add_node(
            "N1".to_string(),
            [s1_in.clone(), s1_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();

    // Create TreeTNs
    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let mpo2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    // True site indices
    let true_s0 = make_index(2);
    let true_s1 = make_index(2);

    // Create LinearOperators
    let lin_op1 = create_linear_operator_from_treetn(
        mpo1,
        &[("N0".to_string(), true_s0.clone(), s0_in.clone())],
        &[("N0".to_string(), true_s0.clone(), s0_out.clone())],
    );

    let lin_op2 = create_linear_operator_from_treetn(
        mpo2,
        &[("N1".to_string(), true_s1.clone(), s1_in.clone())],
        &[("N1".to_string(), true_s1.clone(), s1_out.clone())],
    );

    // No gaps
    let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

    // Compose
    let composed =
        compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
            .expect("Composition should succeed");

    // Verify
    assert_eq!(composed.node_names().len(), 2);
    assert!(composed.get_input_mapping(&"N0".to_string()).is_some());
    assert!(composed.get_input_mapping(&"N1".to_string()).is_some());
}

#[test]
fn test_compose_exclusive_linear_operators_overlap_error() {
    // Target: N0 -- N1 -- N2
    // Op1 covers N0, N1
    // Op2 covers N1, N2 (overlaps at N1!)
    let target = create_chain_site_network(3);

    // Create overlapping networks
    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s1_in = make_index(2);
    op1_net
        .add_node(
            "N0".to_string(),
            [s0_in.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    op1_net
        .add_node(
            "N1".to_string(),
            [s1_in.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    op1_net
        .add_edge(&"N0".to_string(), &"N1".to_string())
        .unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s1_in2 = make_index(2);
    let s2_in = make_index(2);
    op2_net
        .add_node(
            "N1".to_string(),
            [s1_in2.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    op2_net
        .add_node(
            "N2".to_string(),
            [s2_in.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();
    op2_net
        .add_edge(&"N1".to_string(), &"N2".to_string())
        .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let mpo2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    let lin_op1 = LinearOperator::new(mpo1, HashMap::new(), HashMap::new());
    let lin_op2 = LinearOperator::new(mpo2, HashMap::new(), HashMap::new());

    let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

    // Composition should fail due to overlap
    let result =
        compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices);
    assert!(result.is_err(), "Should fail for overlapping operators");
}

#[test]
fn test_compose_gap_identity_tensor_is_diagonal() {
    // Test that gap nodes get proper identity tensors
    // Target: N0 -- N1 -- N2
    // Op covers N0, N2
    // Gap: N1
    let target = create_chain_site_network(3);

    // Create a two-node operator (non-contiguous in target, but we handle this separately)
    // Actually, for exclusivity check, we need connected subtrees.
    // Let's use single-node operators instead.

    let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    op1_net
        .add_node(
            "N0".to_string(),
            [s0_in.clone(), s0_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();

    let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s2_in = make_index(2);
    let s2_out = make_index(2);
    op2_net
        .add_node(
            "N2".to_string(),
            [s2_in.clone(), s2_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo1 = random_treetn::<f64, _, _>(&mut rng, &op1_net, link_space.clone()).unwrap();
    let mpo2 = random_treetn::<f64, _, _>(&mut rng, &op2_net, link_space.clone()).unwrap();

    let true_s0 = make_index(2);
    let true_s2 = make_index(2);

    let lin_op1 = create_linear_operator_from_treetn(
        mpo1,
        &[("N0".to_string(), true_s0.clone(), s0_in.clone())],
        &[("N0".to_string(), true_s0.clone(), s0_out.clone())],
    );

    let lin_op2 = create_linear_operator_from_treetn(
        mpo2,
        &[("N2".to_string(), true_s2.clone(), s2_in.clone())],
        &[("N2".to_string(), true_s2.clone(), s2_out.clone())],
    );

    // Gap for N1 with dimension 3 (to distinguish from operator sites)
    let true_s1_in = make_index(3);
    let true_s1_out = make_index(3);
    let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
    gap_site_indices.insert(
        "N1".to_string(),
        vec![(true_s1_in.clone(), true_s1_out.clone())],
    );

    let composed =
        compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
            .expect("Composition should succeed");

    // Get the tensor at N1 (should be identity with dummy links)
    let n1_idx = composed.mpo().node_index(&"N1".to_string()).unwrap();
    let n1_tensor = composed.mpo().tensor(n1_idx).unwrap();

    // N1 has 2 neighbors (N0 and N2), so 2 dummy links of dim 1 are added
    // Shape should be [3, 3, 1, 1] (site_in, site_out, dummy_link1, dummy_link2)
    // The order may vary, but total size should be 3*3*1*1 = 9
    let n1_dims = n1_tensor.dims();
    let total_size: usize = n1_dims.iter().product();
    assert_eq!(
        total_size, 9,
        "Total tensor size should be 9 (3x3 identity with dim-1 links)"
    );

    // Check that site dimensions 3 appear exactly twice
    let dim3_count = n1_dims.iter().filter(|&&d| d == 3).count();
    assert_eq!(
        dim3_count, 2,
        "Should have exactly 2 site dimensions of size 3"
    );

    // Check it's effectively an identity (only diagonal elements are non-zero)
    let data = n1_tensor.to_vec::<f64>().expect("Expected DenseF64");

    // For identity with dim-1 dummy links, data is still [1,0,0, 0,1,0, 0,0,1]
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "Identity tensor element {} mismatch: {} vs {}",
            i,
            got,
            want
        );
    }
}
